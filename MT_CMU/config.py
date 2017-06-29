emb_size=192 #Embedding size (assumed equal for decoder and encoder)
layer_depth=1 #For the future, currently this is always 1 in the code.
hidden_size=384 #Encoder and Decoder state size (assumed equal)
NUM_EPOCHS=6 #Generally about 10 epochs are sufficient. Suggested method is to pick model with best validation perplexity.
start=0 #start token id
unk=1 #unk token id
stop=2 #stop token id
garbage=3 #garbage (PAD) token id
use_reverse=True #Use bidirectional encoder (or not)
init_mixed=True  #Initialize decoder state with avg of forward and backward encoder. If False, either only the forward or backward state is used for init
init_enc=False #Use the forward encoder state for initializing decoder. If false the backward encoder (a.k.a revcoder) is used.
use_attention=True #Turn attention on or off.

optimizer_type="ADAM" # other option:"SGD"
share_embeddings=False #Share encoder and decoder embeddings
use_downstream=True #Whether to use context vector in final vocab softmax on decoder side.
mode="train" #train,trial,inference Note: in train mode, supply only modelName, in inference mode, supply exact checkpoint path.
problem="SUM" #Abstractive SUM/ De-to-En-MT Note: Summarization data not available on repo.
srcMasking=True #Whether src side masking is on (True recommended)

if problem=="MT":
    min_src_frequency=1 #Minimum frequency to not be UNKed on src side
    min_tgt_frequency=1 #Minimum frequency to not be UNKed on tgt side
    max_train_sentences=99412 #Maximum training sentences
    MAX_SEQ_LEN=200 #Maximum Length (set to high value to make unimportant)
    model_dir="MT_checkpoints/" #Directory to save
    #TGT_LEN_LIMIT=1000
    normalizeLoss=False #Whether to normalize loss per minibatch (recommended False)
    decoder_prev_random=True #Not used yet.
    PRINT_STEP=500 #Print every x minibatches
    batch_size=32 #Batch size

elif problem=="SUM":
    min_src_frequency=15
    min_tgt_frequency=5
    max_train_sentences=500000
    MAX_SEQ_LEN=200
    model_dir="SUM_checkpoints/"
    #TGT_LEN_LIMIT=100
    normalizeLoss=False
    PRINT_STEP=10
    batch_size=64

mem_optimize=True #Memory optimizations (deleting local variables in advance etc)
cudnnBenchmark=True #CuDNN benchmark (purpoted speedup)
