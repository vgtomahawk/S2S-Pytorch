import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
import config as cnfg
if cnfg.problem=="MT":
    import readData as readData
    srcLang="de"
    tgtLang="en"
elif cnfg.problem=="SUM":
    import readDataSum as readData
    srcLang="article"
    tgtLang="title"

import torch_utils as torch_utils
import numpy as np
import random
import math
import datetime
import gc
import sys

class SeqToSeqAttn(nn.Module):
    def __init__(self,cnfg,wids_src=None,wids_tgt=None):

        super(SeqToSeqAttn,self).__init__()
        self.wids_src=wids_src
        self.wids_tgt=wids_tgt
        self.reverse_wids_src=torch_utils.reverseDict(wids_src)
        self.reverse_wids_tgt=torch_utils.reverseDict(wids_tgt)
        self.cnfg=cnfg
        self.cnfg.srcVocabSize=len(self.wids_src)
        self.cnfg.tgtVocabSize=len(self.wids_tgt)
        self.srcEmbeddings=nn.Embedding(self.cnfg.srcVocabSize,self.cnfg.emb_size)
        if self.cnfg.share_embeddings:
            self.tgtEmbeddings=self.srcEmbeddings
        else:
            self.tgtEmbeddings=nn.Embedding(self.cnfg.tgtVocabSize,self.cnfg.emb_size)
        self.encoder=nn.LSTM(self.cnfg.emb_size,self.cnfg.hidden_size)
        if self.cnfg.use_reverse:
            self.revcoder=nn.LSTM(self.cnfg.emb_size,self.cnfg.hidden_size)
        if self.cnfg.use_attention:
            self.decoder=nn.LSTM(self.cnfg.emb_size+self.cnfg.hidden_size,self.cnfg.hidden_size)
        else:
            self.decoder=nn.LSTM(self.cnfg.emb_size,self.cnfg.hidden_size)
        if self.cnfg.use_attention and self.cnfg.use_downstream:
            self.W=nn.Linear(2*self.cnfg.hidden_size,self.cnfg.tgtVocabSize)
        else:
            self.W=nn.Linear(self.cnfg.hidden_size,self.cnfg.tgtVocabSize)


    def getIndex(self,row,inference=False):
        tensor=torch.LongTensor(row)
        if torch.cuda.is_available():
            tensor=tensor.cuda()
        return autograd.Variable(tensor,volatile=inference)

    def init_hidden(self,batch):
        hiddenElem1=torch.zeros(1,batch.shape[1],self.cnfg.hidden_size)
        hiddenElem2=torch.zeros(1,batch.shape[1],self.cnfg.hidden_size)
        if torch.cuda.is_available():
            hiddenElem1,hiddenElem2=hiddenElem1.cuda(),hiddenElem2.cuda()
        return (autograd.Variable(hiddenElem1),autograd.Variable(hiddenElem2))

    def save_checkpoint(self,modelName,optimizer):
        checkpoint={'state_dict':self.state_dict(),'optimizer':optimizer.state_dict()}
        torch.save(checkpoint,self.cnfg.model_dir+modelName+".ckpt")
        print "Saved Model"
        return

    def load_from_checkpoint(self,modelName,optimizer=None):
        checkpoint=torch.load(modelName)
        self.load_state_dict(checkpoint['state_dict'])
        if optimizer!=None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print "Loaded Model"
        return
    
    #def load_checkpoint(filename):

    def decodeAll(self,srcBatches,modelName,method="greedy",evalMethod="BLEU",suffix="test"):
        tgtStrings=[]
        tgtTimes=[]
        totalTime=0.0
        print "Decoding Start Time:",datetime.datetime.now()
        for srcBatch in srcBatches:
            tgtString=None
            startTime=datetime.datetime.now()
            if method=="greedy":
                tgtString=self.greedyDecode(srcBatch)
            endTime=datetime.datetime.now()
            timeTaken=(endTime-startTime).total_seconds()
            totalTime+=timeTaken
            tgtTimes.append(timeTaken)
            tgtStrings.append(tgtString)
        print "Decoding End Time:",datetime.datetime.now()
        print "Total Decoding Time:",totalTime
        
        #Dump Output
        outFileName=modelName+"."+suffix+".output"
        outFile=open(outFileName,"w")
        for tgtString in tgtStrings:
            outFile.write(tgtString+"\n")
        outFile.close()

        #Dump Times
        timeFileName=modelName+"."+suffix+".time"
        timeFile=open(timeFileName,"w")
        for tgtTime in tgtTimes:
            timeFile.write(str(tgtTime)+"\n")
        timeFile.close()

        if evalMethod=="BLEU":
            import os
            BLEUOutput=os.popen("perl multi-bleu.perl -lc "+"en-de/test.en-de.low.en"+" < "+outFileName).read()
            print BLEUOutput
        #Compute BLEU
        elif evalMethod=="ROUGE":
            print "To implement ROUGE"

        return tgtStrings

    def greedyDecode(self,srcBatch):
        #Note: srcBatch is of size 1
        srcSentenceLength=srcBatch.shape[1]
        srcBatch=srcBatch.T
        self.enc_hidden=self.init_hidden(srcBatch)
        enc_out=None
        encoderOuts=[]
        if self.cnfg.use_reverse:
            self.rev_hidden=self.init_hidden(srcBatch)
            rev_out=None
            revcoderOuts=[]

        srcEmbedsSeq=[]
        for rowId,row in enumerate(srcBatch):
            srcEmbeds=self.srcEmbeddings(self.getIndex(row,inference=True))
            if self.cnfg.use_reverse:
                srcEmbedsSeq.append(srcEmbeds)
            enc_out,self.enc_hidden=self.encoder(srcEmbeds.view(1,srcBatch.shape[1],-1),self.enc_hidden)
            encoderOuts.append(enc_out.view(1,-1))

        if self.cnfg.use_reverse:
            srcEmbedsSeq.reverse()
            for srcEmbeds in srcEmbedsSeq:
                rev_out,self.rev_hidden=self.revcoder(srcEmbeds.view(1,srcBatch.shape[1],-1),self.rev_hidden)
                revcoderOuts.append(rev_out.view(1,-1))
            revcoderOuts.reverse()

        #encoderOuts=[encoderOut.view(srcBatch.shape[1],self.cnfg.hidden_size) for encoderOut in encoderOuts]
        if self.cnfg.use_reverse:
            #revcoderOuts=[revcoderOut.view(srcBatch.shape[1],self.cnfg.hidden_size) for revcoderOut in revcoderOuts]
            encoderOuts=[torch.add(x,y) for x,y in zip(encoderOuts,revcoderOuts)]

        if self.cnfg.mem_optimize:
            if self.cnfg.use_reverse:
                del revcoderOuts
                del rev_out
            del srcEmbedsSeq
            del srcBatch
            del enc_out

        #c_0=torch.zeros(encoderOuts[-1].size())
        zeroInit=torch.zeros(encoderOuts[-1].size())
        if torch.cuda.is_available():
            zeroInit=zeroInit.cuda()
        c_0=autograd.Variable(zeroInit)


        self.hidden=self.enc_hidden
        if self.cnfg.use_reverse:
            if self.cnfg.init_mixed==False:
                if self.cnfg.init_enc:
                    self.hidden=self.enc_hidden
                else:
                    self.hidden=self.rev_hidden
            else:
                self.hidden=(torch.add(self.enc_hidden[0],self.rev_hidden[0]),torch.add(self.enc_hidden[1],self.rev_hidden[1]))
                #self.hidden=(torch.mul(torch.add(self.enc_hidden[0],self.rev_hidden[0]),0.5),torch.mul(torch.add(self.enc_hidden[1],self.rev_hidden[1]),0.5))
        
        tgts=[] 
        row=np.array([self.cnfg.start,]*1)
        tgtEmbeds=self.tgtEmbeddings(self.getIndex(row,inference=True))

        if self.cnfg.use_attention:
            tgtEmbeds=torch.cat([tgtEmbeds,c_0],1)
        out,self.hidden=self.decoder(tgtEmbeds.view(1,1,-1),self.hidden)
        out=out.view(1,-1)
        if self.cnfg.use_attention:
            scores=self.W(torch.cat([out,c_0],1))
        else:
            scores=self.W(out)

        maxValues,argmaxes=torch.max(scores,1)
        argmaxValue=argmaxes.view(1).cpu().data.numpy()[0]
        tgts.append(argmaxValue)

        if self.cnfg.mem_optimize:
            del c_0
            del self.enc_hidden
            if self.cnfg.use_reverse:
                del self.rev_hidden
 
        while argmaxValue!=self.cnfg.stop and len(tgts)<2*srcSentenceLength+10: #self.cnfg.TGT_LEN_LIMIT:
            if self.cnfg.use_attention:
                o_t=out
                alphas=F.softmax(torch.cat([torch.sum(encoderOut*o_t,1) for encoderOut in encoderOuts],1))
                encOutTensor=torch.cat([encoderOut.view(1,1,self.cnfg.hidden_size) for encoderOut in encoderOuts],1)
                alphaTensor=(torch.unsqueeze(alphas,2)).expand(encOutTensor.size())
            
            if self.cnfg.mem_optimize:
                #Free useless references before large dot product
                if self.cnfg.use_attention:
                    del alphas
                    del o_t

            if self.cnfg.use_attention:
                c_t=torch.squeeze(torch.sum(alphaTensor*encOutTensor,1)).view(1,-1)

             
            row=np.array([argmaxValue,]*1)
            tgtEmbeds=self.tgtEmbeddings(self.getIndex(row,inference=True))
            #print tgtEmbeds.size()
            #print c_t.size()

            if self.cnfg.use_attention:
                tgtEmbeds=torch.cat([tgtEmbeds,c_t],1).view(1,1,-1)
            else:
                tgtEmbeds=tgtEmbeds.view(1,1,-1)

            out,self.hidden=self.decoder(tgtEmbeds,self.hidden)
            out=out.view(1,-1)
            if self.cnfg.use_attention:
                scores=self.W(torch.cat([out,c_t],1))
            else:
                scores=self.W(out)

            maxValues,argmaxes=torch.max(scores,1)
            argmaxValue=argmaxes.view(1).cpu().data.numpy()[0]
            tgts.append(argmaxValue)

        if tgts[-1]==self.cnfg.stop:
            tgts=tgts[:-1]

        return " ".join([self.reverse_wids_tgt[x] for x in tgts])

    
    def forward(self,srcBatch,batch,srcMask,mask,inference=False):
        srcBatch=srcBatch.T
        srcMask=srcMask.T
        #Init encoder. We don't need start here since we don't softmax.
        self.enc_hidden=self.init_hidden(srcBatch)
        enc_out=None
        encoderOuts=[]

        if self.cnfg.use_reverse:
            self.rev_hidden=self.init_hidden(srcBatch)
            rev_out=None
            revcoderOuts=[]

        srcEmbedsSeq=[]
        for rowId,row in enumerate(srcBatch):
            srcEmbeds=self.srcEmbeddings(self.getIndex(row,inference=inference))
            if self.cnfg.use_reverse:
                srcEmbedsSeq.append(srcEmbeds)
            enc_out,self.enc_hidden=self.encoder(srcEmbeds.view(1,srcBatch.shape[1],-1),self.enc_hidden)
            encoderOuts.append(enc_out.squeeze())

        if self.cnfg.use_reverse:
            srcEmbedsSeq.reverse()
            for srcEmbeds in srcEmbedsSeq:
                rev_out,self.rev_hidden=self.revcoder(srcEmbeds.view(1,srcBatch.shape[1],-1),self.rev_hidden)
                revcoderOuts.append(rev_out.squeeze())
            revcoderOuts.reverse()

        #encoderOuts=[encoderOut.view(srcBatch.shape[1],self.cnfg.hidden_size) for encoderOut in encoderOuts]
        if self.cnfg.use_reverse:
            #revcoderOuts=[revcoderOut.view(srcBatch.shape[1],self.cnfg.hidden_size) for revcoderOut in revcoderOuts]
            encoderOuts=[torch.add(x,y) for x,y in zip(encoderOuts,revcoderOuts)]

        if self.cnfg.srcMasking:
            srcMaskTensor=torch.Tensor(srcMask)
            if torch.cuda.is_available():
                srcMaskTensor=srcMaskTensor.cuda()
            srcMaskTensor=torch.chunk(autograd.Variable(srcMaskTensor),len(encoderOuts),0)
            srcMaskTensor=[x.view(-1,1) for x in srcMaskTensor]
            #print encoderOuts[0].size()
            encoderOuts=[encoderOut*(x.expand(encoderOut.size())) for encoderOut,x in zip(encoderOuts,srcMaskTensor)]
            #print encoderOuts[0].size()
            del srcMaskTensor

        if self.cnfg.mem_optimize:
            if self.cnfg.use_reverse:
                del revcoderOuts
                del rev_out
            del srcEmbedsSeq
            del srcBatch
            del enc_out

        zeroInit=torch.zeros(encoderOuts[-1].size())
        if torch.cuda.is_available():
            zeroInit=zeroInit.cuda()
        c_0=autograd.Variable(zeroInit)


        batch=batch.T
        mask=mask.T
        if torch.cuda.is_available():
            maskTensor=autograd.Variable(torch.Tensor(mask).cuda())
        else:
            maskTensor=autograd.Variable(torch.Tensor(mask))


        self.hidden=self.enc_hidden
        if self.cnfg.use_reverse:
            if self.cnfg.init_mixed==False:
                if self.cnfg.init_enc:
                    self.hidden=self.enc_hidden
                else:
                    self.hidden=self.rev_hidden
            else:
                #self.hidden=(torch.mul(torch.add(self.enc_hidden[0],self.rev_hidden[0]),0.5),torch.mul(torch.add(self.enc_hidden[1],self.rev_hidden[1]),0.5))
                self.hidden=(torch.add(self.enc_hidden[0],self.rev_hidden[0]),torch.add(self.enc_hidden[1],self.rev_hidden[1]))

        
        #Init with START token
        if self.cnfg.use_attention:
            contextVectors=[]
            contextVectors.append(c_0)
        row=np.array([self.cnfg.start,]*batch.shape[1])
        tgtEmbeds=self.tgtEmbeddings(self.getIndex(row,inference=inference))
        if self.cnfg.use_attention:
            tgtEmbeds=torch.cat([tgtEmbeds,c_0],1)
        out,self.hidden=self.decoder(tgtEmbeds.view(1,batch.shape[1],-1),self.hidden)
        if self.cnfg.mem_optimize:
            del c_0
            del self.enc_hidden
            if self.cnfg.use_reverse:
                del self.rev_hidden
       

        decoderOuts=[out.squeeze(),]
        tgts=[]
        for rowId,row in enumerate(batch):
            
            if self.cnfg.use_attention:
                o_t=decoderOuts[-1]
                alphas=F.softmax(torch.cat([torch.sum(encoderOut*o_t,1) for encoderOut in encoderOuts],1))
                encOutTensor=torch.cat([encoderOut.view(batch.shape[1],1,self.cnfg.hidden_size) for encoderOut in encoderOuts],1)
                alphaTensor=(torch.unsqueeze(alphas,2)).expand(encOutTensor.size())
            
            if self.cnfg.mem_optimize:
                #Free useless references before large dot product
                if self.cnfg.use_attention:
                    del alphas
                    del o_t

            if self.cnfg.use_attention:
                c_t=torch.squeeze(torch.sum(alphaTensor*encOutTensor,1))
                contextVectors.append(c_t) 

            tgtEmbeds=self.tgtEmbeddings(self.getIndex(row,inference=inference))
            tgts.append(self.getIndex(row))
            if self.cnfg.use_attention:
                tgtEmbeds=torch.cat([tgtEmbeds,c_t],1).view(1,batch.shape[1],-1)
            else:
                tgtEmbeds=tgtEmbeds.view(1,batch.shape[1],-1)
            out,self.hidden=self.decoder(tgtEmbeds,self.hidden)
            #decoderOuts.append(out.view(batch.shape[1],self.cnfg.hidden_size))
            decoderOuts.append(out.squeeze())
            if self.cnfg.mem_optimize:
                if self.cnfg.use_attention:
                    del alphaTensor,encOutTensor,c_t,tgtEmbeds
                else:
                    del tgtEmbeds

        decoderOuts=decoderOuts[:-1]
        if self.cnfg.use_attention:
            contextVectors=contextVectors[:-1]

        if self.cnfg.use_attention and self.cnfg.use_downstream:
            decoderOuts=[torch.cat([decoderOut,c_t],1) for decoderOut,c_t in zip(decoderOuts,contextVectors)]
        
        if self.cnfg.mem_optimize:
            del encoderOuts
            del out
            del self.hidden
            if self.cnfg.use_attention:
                del contextVectors
            gc.collect()

        decoderOuts=[F.log_softmax(self.W(decoderOut)) for decoderOut in decoderOuts]

        decoderOuts=torch.cat(decoderOuts,0)
        tgts=torch.cat(tgts,0)

    
        tgts=tgts.view(-1,1)
        decoderOuts=-torch.gather(decoderOuts,dim=1,index=tgts)
        maskTensor=maskTensor.view(decoderOuts.size())
        decoderOuts=decoderOuts*maskTensor
        
        if self.cnfg.normalizeLoss==True:
            totalLoss=decoderOuts.sum()/maskTensor.sum()
        else:
            totalLoss=decoderOuts.sum()
            

        return totalLoss



def main():
    torch.manual_seed(1)
    random.seed(7867567)

    modelName=sys.argv[1]

    wids_src=defaultdict(lambda: len(wids_src))
    wids_tgt=defaultdict(lambda: len(wids_tgt))

    
    train_src=readData.read_corpus(wids_src,mode="train",update_dict=True,min_frequency=cnfg.min_src_frequency,language=srcLang)
    train_tgt=readData.read_corpus(wids_tgt,mode="train",update_dict=True,min_frequency=cnfg.min_tgt_frequency,language=tgtLang)

    valid_src=readData.read_corpus(wids_src,mode="valid",update_dict=False,min_frequency=cnfg.min_src_frequency,language=srcLang)
    valid_tgt=readData.read_corpus(wids_tgt,mode="valid",update_dict=False,min_frequency=cnfg.min_tgt_frequency,language=tgtLang)

    test_src=readData.read_corpus(wids_src,mode="test",update_dict=False,min_frequency=cnfg.min_src_frequency,language=srcLang)
    test_tgt=readData.read_corpus(wids_tgt,mode="test",update_dict=False,min_frequency=cnfg.min_tgt_frequency,language=tgtLang)



    train_src,train_tgt=train_src[:cnfg.max_train_sentences],train_tgt[:cnfg.max_train_sentences]
    print "src vocab size:",len(wids_src)
    print "tgt vocab size:",len(wids_tgt)
    print "training size:",len(train_src)
    print "valid size:",len(valid_src)

    train=zip(train_src,train_tgt) #zip(train_src,train_tgt)
    valid=zip(valid_src,valid_tgt) #zip(train_src,train_tgt)
    

    #train.sort(key=lambda x:-len(x[1]))
    #valid.sort(key=lambda x:-len(x[1]))

    train.sort(key=lambda x:len(x[0]))
    valid.sort(key=lambda x:len(x[0]))


    train_src,train_tgt=[x[0] for x in train],[x[1] for x in train]
    valid_src,valid_tgt=[x[0] for x in valid],[x[1] for x in valid]
    

    #NUM_TOKENS=sum([len(x) for x in train_tgt])

    train_src_batches,train_src_masks=torch_utils.splitBatches(train=train_src,batch_size=cnfg.batch_size,padSymbol=cnfg.garbage,method="pre")
    train_tgt_batches,train_tgt_masks=torch_utils.splitBatches(train=train_tgt,batch_size=cnfg.batch_size,padSymbol=cnfg.garbage,method="post")
    valid_src_batches,valid_src_masks=torch_utils.splitBatches(train=valid_src,batch_size=cnfg.batch_size,padSymbol=cnfg.garbage,method="pre")
    valid_tgt_batches,valid_tgt_masks=torch_utils.splitBatches(train=valid_tgt,batch_size=cnfg.batch_size,padSymbol=cnfg.garbage,method="post")
    test_src_batches,test_src_masks=torch_utils.splitBatches(train=test_src,batch_size=1,padSymbol=cnfg.garbage,method="pre")
    test_tgt_batches,test_tgt_masks=torch_utils.splitBatches(train=test_tgt,batch_size=1,padSymbol=cnfg.garbage,method="post")


    #Dump useless references
    train=None
    valid=None
    #Sanity check
    assert (len(train_tgt_batches)==len(train_src_batches))
    assert (len(valid_tgt_batches)==len(valid_src_batches))
    assert (len(test_tgt_batches)==len(test_src_batches))

    print "Training Batches:",len(train_tgt_batches)
    print "Validation Batches:",len(valid_tgt_batches)
    print "Test Points:",len(test_src_batches)

    if cnfg.cudnnBenchmark:
        torch.backends.cudnn.benchmark=True
    #Declare model object
    print "Declaring Model, Loss, Optimizer"
    model=SeqToSeqAttn(cnfg,wids_src=wids_src,wids_tgt=wids_tgt)
    loss_function=nn.NLLLoss()
    if torch.cuda.is_available():
        model.cuda()
        loss_function=loss_function.cuda()
    optimizer=None
    if cnfg.optimizer_type=="SGD":
        optimizer=optim.SGD(model.parameters(),lr=0.05)
    elif cnfg.optimizer_type=="ADAM":
        optimizer=optim.Adam(model.parameters())

    if cnfg.mode=="trial":
        print "Running Sample Batch" 
        print "Source Batch Shape:",train_src_batches[30].shape
        print "Source Mask Shape:",train_src_masks[30].shape
        print "Target Batch Shape:",train_tgt_batches[30].shape
        print "Target Mask Shape:",train_tgt_masks[30].shape
        sample_src_batch=train_src_batches[30]
        sample_tgt_batch=train_tgt_batches[30]
        sample_mask=train_tgt_masks[30]
        sample_src_mask=train_src_masks[30]
        print datetime.datetime.now() 
        model.zero_grad()
        loss=model(sample_src_batch,sample_tgt_batch,sample_src_mask,sample_mask)
        print loss
        loss.backward()
        optimizer.step()
        print datetime.datetime.now()
        #print torch.backends.cudnn.benchmark
        #print torch.backends.cudnn.enabled
        print "Done Running Sample Batch"

    train_batches=zip(train_src_batches,train_tgt_batches,train_src_masks,train_tgt_masks)
    valid_batches=zip(valid_src_batches,valid_tgt_batches,valid_src_masks,valid_tgt_masks)

    train_src_batches,train_tgt_batches,train_src_masks,train_tgt_masks=None,None,None,None
    valid_src_batches,valid_tgt_batches,valid_src_masks,valid_tgt_masks=None,None,None,None
    if cnfg.mode=="train":
        print "Start Time:",datetime.datetime.now()     
        for epochId in range(cnfg.NUM_EPOCHS):
            random.shuffle(train_batches)
            for batchId,batch in enumerate(train_batches):
                src_batch,tgt_batch,src_mask,tgt_mask=batch[0],batch[1],batch[2],batch[3]
                batchLength=src_batch.shape[1]
                batchSize=src_batch.shape[0]
                #print "Batch Length:",batchLength
                if batchLength<cnfg.MAX_SEQ_LEN and batchSize>1:
                    model.zero_grad()
                    loss=model(src_batch,tgt_batch,src_mask,tgt_mask)
                    if cnfg.mem_optimize:
                        del src_batch,tgt_batch,src_mask,tgt_mask
                    loss.backward()
                    if cnfg.mem_optimize:
                        del loss
                    optimizer.step()               
                if batchId%cnfg.PRINT_STEP==0:
                    print "Batch No:",batchId," Time:",datetime.datetime.now()

            totalValidationLoss=0.0
            NUM_TOKENS=0.0
            for batchId,batch in enumerate(valid_batches):
                src_batch,tgt_batch,src_mask,tgt_mask=batch[0],batch[1],batch[2],batch[3]
                model.zero_grad()
                loss=model(src_batch,tgt_batch,src_mask,tgt_mask,inference=True)
                if cnfg.normalizeLoss:
                    totalValidationLoss+=(loss.data.cpu().numpy())*np.sum(tgt_mask)
                else:
                    totalValidationLoss+=(loss.data.cpu().numpy())
                NUM_TOKENS+=np.sum(tgt_mask)
                if cnfg.mem_optimize:
                    del src_batch,tgt_batch,src_mask,tgt_mask,loss
            
            model.save_checkpoint(modelName+"_"+str(epochId),optimizer)

            perplexity=math.exp(totalValidationLoss/NUM_TOKENS)
            print "Epoch:",epochId," Total Validation Loss:",totalValidationLoss," Perplexity:",perplexity
        print "End Time:",datetime.datetime.now()

    elif cnfg.mode=="inference":
        model.load_from_checkpoint(modelName)
        #print " ".join([model.reverse_wids_src[x] for x in test_src_batches[1][0]])
        #print " ".join([model.reverse_wids_tgt[x] for x in test_tgt_batches[1][0]])
        #model(test_src_batches[1],test_tgt_batches[1],test_tgt_masks[1])      
      
        model.decodeAll(test_src_batches,modelName,method="greedy",evalMethod="BLEU",suffix="test")

main()
