emb_size=192
layer_depth=1
hidden_size=384
NUM_EPOCHS=6
start=0
unk=1
stop=2
garbage=3
use_reverse=True
init_mixed=True
init_enc=False
use_attention=True

optimizer_type="ADAM" # other option:"SGD"
share_embeddings=False
use_downstream=True
mode="train" #train,trial,inference
problem="SUM" #SUM/MT
srcMasking=True

if problem=="MT":
    min_src_frequency=1
    min_tgt_frequency=1
    max_train_sentences=99412
    MAX_SEQ_LEN=200
    model_dir="MT_checkpoints/"
    #TGT_LEN_LIMIT=1000
    normalizeLoss=False
    decoder_prev_random=True
    PRINT_STEP=500
    batch_size=32

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

mem_optimize=True
cudnnBenchmark=True
