# GLOBAL
NUM_POINTS = 256
FEATURE_OUTPUT_DIM = 512
PICKLE_FOLDER = "train_pickle/"
RESULTS_FOLDER = "results/"
OUTPUT_FILE = "results/results.txt"
SIZED_GRID_X = 64*4
SIZED_GRID_Y = 64
GRID_X = 1080
GRID_Y = 1920
file_name = "Goffs"

LOG_DIR = 'log/'
MODEL_FILENAME = "model.ckpt"

DATASET_FOLDER = '/home/cc/dm_data'
DATASET_FOLDER_RGB = '/mnt/NAS/home/cc/data/habitat_4/train'
DATASET_FOLDER_RGB_REAL = '/mnt/NAS/data/cc_data/2D_RGB_real_full3'

# TRAIN
BATCH_NUM_QUERIES = 2
TRAIN_POSITIVES_PER_QUERY = 2
TRAIN_NEGATIVES_PER_QUERY = 18
DECAY_STEP = 200000
DECAY_RATE = 0.7
BASE_LEARNING_RATE = 0.000005
MOMENTUM = 0.9
OPTIMIZER = 'ADAM'
MAX_EPOCH = 20
FOLD_NUM = 18

MARGIN_1 = 0.5
MARGIN_2 = 0.2

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_CLIP = 0.99

RESUME = False
ROT_NUM = 8

EVAL_NEAREST = 16
INIT_TRUST = 2
INIT_TRUST_SCALAR = 1
NEIGHBOR = 4

TRAIN_FILE = 'generating_queries/training_queries_baseline.pickle'
TEST_FILE = 'generating_queries/test_queries_baseline.pickle'
scene_list = ['Goffs','Nimmons','Reyno','Spotswood','Springhill','Stilwell']

# LOSS
LOSS_FUNCTION = 'quadruplet'
LOSS_FUNCTION_RGB = 'triplet'

LOSS_LAZY = True
TRIPLET_USE_BEST_POSITIVES = False
LOSS_IGNORE_ZERO_BATCH = False

# EVAL6
EVAL_BATCH_SIZE = 2
EVAL_POSITIVES_PER_QUERY = 4
EVAL_NEGATIVES_PER_QUERY = 12

EVAL_DATABASE_FILE = 'generating_queries/evaluation_database.pickle'
EVAL_QUERY_FILE = 'generating_queries/evaluation_query.pickle'


def cfg_str():
    out_string = ""
    for name in globals():
        if not name.startswith("__") and not name.__contains__("cfg_str"):
            #print(name, "=", globals()[name])
            out_string = out_string + "cfg." + name + \
                "=" + str(globals()[name]) + "\n"
    return out_string
