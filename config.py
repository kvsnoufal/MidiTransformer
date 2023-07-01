DEVICE ='cuda' #'cuda'
EPOCHS = 10
BATCH_SIZE = 1900
SEQ_LEN = 64
VOCAB_SIZE = 663
LOG_DIR = 'output/logs'
TRAIN_TEST_RATIO = 0.9

EMBED_DIM = 768
TRANSFORMER_HEADS = 8
TRANSFORMER_LAYERS = 1
LR = 9e-6
SUFFIX="adlremiplus_small_9e6"

MODEL_DIR = 'output/model_{}'.format(SUFFIX)
ENCODING_DIR = '../input/adl_encodings_rplus'
MIDI_DIR = "../input/adl-piano-midi"