import sys
from collections import defaultdict

def read_corpus(wids,mode="train",update_dict=True,min_frequency=3,language="article"):
    filePath="/data/vgangal/NAMAS/working_dir/"
    fileName=None
    if mode=="train":
        fileName="train."+language+".txt"
    elif mode=="valid":
        fileName="valid."+language+".txt"
    elif mode=="test":
        fileName="test."+language+".txt"
    fileName=filePath+fileName

    
    if update_dict:
        wids["<s>"]=0
        wids["<unk>"]=1
        wids["</s>"]=2
        wids["<GARBAGE>"]=3

        word_frequencies=defaultdict(int)
        for line in open(fileName):
            words=line.split()
            for word in words:
                word_frequencies[word]+=1
        
        for word,freq in word_frequencies.items():
            if freq>min_frequency:
                wids[word]=len(wids)

    sentences=[]

    for line in open(fileName):
        words=line.split()
        #words.insert(0,"<s>")
        words.append("</s>")
        sentence=[wids[word] if word in wids else 1 for word in words]    
        sentences.append(sentence)

    return sentences
            



if __name__=="__main__":
    wids=defaultdict(lambda: len(wids))
    train_sentences=read_corpus(wids,mode="train",update_dict=True,language="title")
    valid_sentences=read_corpus(wids,mode="valid",update_dict=False,language="title")
    test_sentences=read_corpus(wids,mode="test",update_dict=False,language="title")
    print len(wids)
    print len(train_sentences)
    print len(valid_sentences)
    print len(test_sentences)
