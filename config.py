SEED = 42

N_STUDENTS = 48
N_DAYS = 84

TRAIN_IDS = list(range(38))
TEST_IDS = list(range(38, 48))

SEQ_LEN = 8
BATCH_SIZE = 128

EPOCHS = 30
LR = 3e-4

HIDDEN_DIM = 128
N_CLASSES = 3

CLASS_NAMES = [
    "negative",
    "neutral",
    "positive"
]