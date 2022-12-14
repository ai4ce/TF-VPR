import datetime
#import torch
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse
import importlib
import math
import os
import socket
import sys

import numpy as np
from sklearn.neighbors import KDTree, NearestNeighbors
import generating_queries.generate_training_tuples_RGB_real_baseline as generate_dataset_tt
import generating_queries.generate_test_RGB_real_baseline_sets as generate_dataset_eval

import config as cfg
import evaluate
import loss.pointnetvlad_loss as PNV_loss
import models.Verification_RGB_real as VFC
import models.ImageNetVlad as INV
import torch
import torch.nn as nn
from loading_pointclouds import *
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.backends import cudnn
import scipy.io as sio

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)



cudnn.enabled = True

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default='log/', help='Log dir [default: log]')
parser.add_argument('--results_dir', default='results/',
                    help='results dir [default: results]')
parser.add_argument('--positives_per_query', type=int, default=2,
                    help='Number of potential positives in each training tuple [default: 2]')
parser.add_argument('--negatives_per_query', type=int, default=18,
                    help='Number of definite negatives in each training tuple [default: 18]')
parser.add_argument('--max_epoch', type=int, default=100,
                    help='Epoch to run [default: 100]')
parser.add_argument('--batch_num_queries', type=int, default=2,
                    help='Batch Size during training [default: 2]')
parser.add_argument('--learning_rate', type=float, default=0.000005,
                    help='Initial learning rate [default: 0.000005]')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam',
                    help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000,
                    help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7,
                    help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--margin_1', type=float, default=0.5,
                    help='Margin for hinge loss [default: 0.5]')
parser.add_argument('--margin_2', type=float, default=0.2,
                    help='Margin for hinge loss [default: 0.2]')
parser.add_argument('--loss_function', default='triplet', choices=[
                    'triplet', 'quadruplet'], help='triplet or quadruplet [default: quadruplet]')
parser.add_argument('--loss_not_lazy', action='store_false',
                    help='If present, do not use lazy variant of loss')
parser.add_argument('--loss_ignore_zero_batch', action='store_true',
                    help='If present, mean only batches with loss > 0.0')
parser.add_argument('--triplet_use_best_positives', action='store_true',
                    help='If present, use best positives, otherwise use hardest positives')
parser.add_argument('--resume', action='store_true',
                    help='If present, restore checkpoint and resume training')
parser.add_argument('--dataset_folder', default='/mnt/NAS/data/cc_data',
                    help='PointNetVlad Dataset Folder')

FLAGS = parser.parse_args()
#cfg.EVAL_BATCH_SIZE = 12
cfg.GRID_X = 1080
cfg.GRID_Y = 1920
cfg.MAX_EPOCH = FLAGS.max_epoch
cfg.BASE_LEARNING_RATE = FLAGS.learning_rate
cfg.MOMENTUM = FLAGS.momentum
cfg.OPTIMIZER = FLAGS.optimizer
cfg.DECAY_STEP = FLAGS.decay_step
cfg.DECAY_RATE = FLAGS.decay_rate
cfg.MARGIN1 = FLAGS.margin_1
cfg.MARGIN2 = FLAGS.margin_2

cfg.TRIPLET_USE_BEST_POSITIVES = FLAGS.triplet_use_best_positives
cfg.LOSS_LAZY = FLAGS.loss_not_lazy
cfg.LOSS_IGNORE_ZERO_BATCH = FLAGS.loss_ignore_zero_batch

cfg.TRAIN_FILE = 'generating_queries/train_pickle/training_queries_baseline_0.pickle'
cfg.TEST_FILE = 'generating_queries/train_pickle/test_queries_baseline_0.pickle'
cfg.DB_FILE = 'generating_queries/train_pickle/db_queries_baseline_0.pickle'

cfg.LOG_DIR = FLAGS.log_dir
if not os.path.exists(cfg.LOG_DIR):
    os.mkdir(cfg.LOG_DIR)
LOG_FOUT = open(os.path.join(cfg.LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

cfg.RESULTS_FOLDER = FLAGS.results_dir
print("cfg.RESULTS_FOLDER:"+str(cfg.RESULTS_FOLDER))

#cfg.DATASET_FOLDER = '/mnt/NAS/home/yiming/habitat/Springhill'

# Load dictionary of training queries
TRAINING_QUERIES = get_queries_dict(cfg.TRAIN_FILE)
TEST_QUERIES = get_queries_dict(cfg.TEST_FILE)
DB_QUERIES = get_queries_dict(cfg.DB_FILE)

cfg.BN_INIT_DECAY = 0.5
cfg.BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(cfg.DECAY_STEP)
cfg.BN_DECAY_CLIP = 0.99

HARD_NEGATIVES = {}
TRAINING_LATENT_VECTORS = []

TOTAL_ITERATIONS = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.margin = 0.1

def get_bn_decay(batch):
    bn_momentum = cfg.BN_INIT_DECAY * \
        (cfg.BN_DECAY_DECAY_RATE **
         (batch * cfg.BATCH_NUM_QUERIES // BN_DECAY_DECAY_STEP))
    return min(cfg.BN_DECAY_CLIP, 1 - bn_momentum)


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)

# learning rate halfed every 5 epoch

def get_learning_rate(epoch):
    learning_rate = cfg.BASE_LEARNING_RATE * ((0.9) ** (epoch // 5))
    learning_rate = max(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate


def train(scene_index):
    train_start = datetime.datetime.now()
    global HARD_NEGATIVES, TOTAL_ITERATIONS, TRAINING_QUERIES
    bn_decay = get_bn_decay(0)
    #tf.summary.scalar('bn_decay', bn_decay)
    '''
    generate_dataset_tt.generate(scene_index, 0, inside=False)
    generate_dataset_eval.generate(scene_index, False, inside=False)
    generate_dataset_eval.generate(scene_index, True, inside=False)
    '''
    TRAINING_QUERIES = get_queries_dict(cfg.TRAIN_FILE)
    TEST_QUERIES = get_queries_dict(cfg.TEST_FILE)
    DB_QUERIES = get_queries_dict(cfg.DB_FILE)

    cfg.RESULTS_FOLDER = os.path.join("results/")
    if not os.path.isdir(cfg.RESULTS_FOLDER):
        os.mkdir(cfg.RESULTS_FOLDER)
    #loss = lazy_quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg_vec, MARGIN1, MARGIN2)
    if cfg.LOSS_FUNCTION_RGB == 'quadruplet':
        loss_function = PNV_loss.quadruplet_loss
    elif cfg.LOSS_FUNCTION_RGB == 'triplet_RI':
        loss_function = PNV_loss.triplet_loss_RI
    else:
        loss_function = PNV_loss.triplet_loss
    learning_rate = get_learning_rate(0)
    
    train_writer = SummaryWriter(os.path.join(cfg.LOG_DIR, 'train'))
    #test_writer = SummaryWriter(os.path.join(cfg.LOG_DIR, 'test'))

    model = INV.ImageNetVlad(global_feat=True, feature_transform=True,
                             max_pool=False, output_dim=cfg.FEATURE_OUTPUT_DIM, grid_x = cfg.GRID_X, grid_y = cfg.GRID_Y)
    model = model.to(device)

    parameters = filter(lambda p: p.requires_grad, model.parameters())

    if cfg.OPTIMIZER == 'momentum':
        optimizer = torch.optim.SGD(
            parameters, learning_rate, momentum=cfg.MOMENTUM)
    elif cfg.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(parameters, learning_rate)
    else:
        optimizer = None
        exit(0)

    if FLAGS.resume:
        resume_filename = cfg.LOG_DIR + "checkpoint.pth.tar"
        print("Resuming From ", resume_filename)
        checkpoint = torch.load(resume_filename)
        saved_state_dict = checkpoint['state_dict']
        starting_epoch = checkpoint['epoch']
        print("starting_epoch:"+str(starting_epoch))
        #starting_epoch = starting_epoch +1
        TOTAL_ITERATIONS = starting_epoch * len(TRAINING_QUERIES)
        starting_epoch = starting_epoch +1
        model.load_state_dict(saved_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        #trusted_positives = sio.loadmat("results/trusted_positives_folder/trusted_positives_"+str(starting_epoch)+".mat")['data']
        #potential_positives = sio.loadmat("results/trusted_positives_folder/potential_positives_"+str(starting_epoch)+".mat")['data']
        #potential_distributions = sio.loadmat("results/trusted_positives_folder/potential_distributions_"+str(starting_epoch)+".mat")['data']
        
        #print("trusted_positives:"+str(trusted_positives.shape))
        #print("potential_positives:"+str(potential_positives.shape))
        #print("potential_distributions:"+str(potential_distributions.shape))
    else:
        starting_epoch = 0

    #model = nn.DataParallel(model)

    LOG_FOUT.write(cfg.cfg_str())
    LOG_FOUT.write("\n")
    LOG_FOUT.flush()

    try:
        potential_positives
    except NameError:
        potential_positives = None
        potential_distributions = None
        trusted_positives = None
    
    #criterion = nn.TripletMarginLoss(margin=cfg.margin**0.5, 
    #                        p=2, reduction='sum').to(device)

    for epoch in range(starting_epoch, cfg.MAX_EPOCH):
        print(epoch)
        print()
        '''
        if trusted_positives is not None:
            sio.savemat("results/trusted_positives_folder/trusted_positives_"+str(epoch)+".mat",{'data':trusted_positives})
            sio.savemat("results/trusted_positives_folder/potential_positives_"+str(epoch)+".mat",{'data':potential_positives})
            sio.savemat("results/trusted_positives_folder/potential_distributions_"+str(epoch)+".mat",{'data':potential_distributions})
       
        generate_dataset_tt.generate(scene_index, epoch, definite_positives=trusted_positives, inside=False)
        
        TRAIN_FILE = 'generating_queries/train_pickle/training_queries_baseline_'+str(epoch)+'.pickle'
        TEST_FILE = 'generating_queries/train_pickle/test_queries_baseline_'+str(epoch)+'.pickle'
        DB_FILE = 'generating_queries/train_pickle/db_queries_baseline_'+str(epoch)+'.pickle'

        TRAINING_QUERIES = get_queries_dict(TRAIN_FILE)
        TEST_QUERIES = get_queries_dict(TEST_FILE)
        DB_QUERIES = get_queries_dict(DB_FILE)
        '''
        log_string('**** EPOCH %03d ****' % (epoch))
        sys.stdout.flush()
        
        train_one_epoch(model, optimizer, train_writer, loss_function, epoch, scene_index, TRAINING_QUERIES, TEST_QUERIES, DB_QUERIES)
        '''
        log_string('EVALUATING...')
        cfg.OUTPUT_FILE = os.path.join(cfg.RESULTS_FOLDER, 'results_' + str(epoch) + '.txt')
        
        db_vec = evaluate.evaluate_model(model,optimizer,epoch,scene_index,True,True)
        
        db_vec = np.array(db_vec)
        print("db_vec:"+str(db_vec.shape))

        db_vec_all = db_vec.reshape(-1,db_vec.shape[-1])
        print("db_vec_all:"+str(db_vec_all.shape))

        nbrs = NearestNeighbors(n_neighbors=cfg.INIT_TRUST, algorithm='ball_tree', n_jobs =18).fit(db_vec_all)
        distance, indice = nbrs.kneighbors(db_vec_all)

        weight = np.exp(-distance*10)
        indice = indice.tolist()
        weight = weight.tolist()
        print("weight:"+str(np.array(weight).shape))
        # assert(0)

        if potential_positives is None:
            potential_positives = []
            potential_distributions = []
            trusted_positives = []
            
            potential_positives = indice
            potential_distributions = weight

            # pool = Pool(processes=db_vec.shape[0])
            # inputs = [(True, db_vec, rank, [], [], None, folders, thresholds, all_files_reshape, weight, indice, epoch) for rank in range(db_vec.shape[0])]
            _, _, trusted_positives = VFC.Compute_positive(True, db_vec, [], [], None, weight, indice, epoch)
            

        else:
            new_potential_positives = []
            new_potential_distributions = []
            new_trusted_positives = []

            # pool = Pool(processes=db_vec.shape[0])
            # inputs = [(False, db_vec, rank, potential_positives, potential_distributions, trusted_positives, thresholds, all_files_reshape, weight, indice, epoch) for rank in range(db_vec.shape[0])]
            potential_positives, potential_distributions, trusted_positives = VFC.Compute_positive(False, db_vec, potential_positives, potential_distributions, trusted_positives, weight, indice, epoch)

            # new_potential_positives.append(potential_positive)
            # new_potential_distributions.append(potential_distribution)
            # new_trusted_positives.append(trusted_positive)
        
            # potential_positives = new_potential_positives
            # potential_distributions = new_potential_distributions
            # trusted_positives = new_trusted_positives
        
        # asdasdasd
        # 
        '''
        log_string('EVALUATING...')
        cfg.OUTPUT_FILE = os.path.join(cfg.RESULTS_FOLDER , 'results_' + str(epoch) + '.txt')
        train_end = datetime.datetime.now()
        eval_start = datetime.datetime.now()
        eval_recall_1, eval_recall_5, eval_recall_10 = evaluate.evaluate_model_RGB_real(model,optimizer,epoch,scene_index,True)
        eval_end = datetime.datetime.now()


        log_string('EVAL RECALL_1: %s' % str(eval_recall_1))
        log_string('EVAL RECALL_5: %s' % str(eval_recall_5))
        log_string('EVAL RECALL_10: %s' % str(eval_recall_10))


def train_one_epoch(model, optimizer, train_writer, loss_function, epoch, scene_index, TRAINING_QUERIES, TEST_QUERIES, DB_QUERIES):
    global HARD_NEGATIVES
    global TRAINING_LATENT_VECTORS, TOTAL_ITERATIONS

    is_training = True
    sampled_neg = 4000
    # number of hard negatives in the training tuple
    # which are taken from the sampled negatives
    num_to_take = 10

    # Shuffle train files
    train_file_idxs = np.arange(0, len(TRAINING_QUERIES.keys()))
    #print("train_file_idxs:"+str(len(train_file_idxs)))
    #assert(0)
    np.random.shuffle(train_file_idxs)
    
    for i in range(len(train_file_idxs)//cfg.BATCH_NUM_QUERIES):
    # for i in range(1):
        batch_keys = train_file_idxs[i *
                                     cfg.BATCH_NUM_QUERIES:(i+1)*cfg.BATCH_NUM_QUERIES]
        q_tuples = []
        
        faulty_tuple = False
        no_other_neg = False
        for j in range(cfg.BATCH_NUM_QUERIES):
            #print("positives:"+str(TRAINING_QUERIES[batch_keys[j]]['positives']))
            #print("negatives:"+str(TRAINING_QUERIES[batch_keys[j]]['negatives']))
            if (len(TRAINING_QUERIES[batch_keys[j]]["positives"]) < cfg.TRAIN_POSITIVES_PER_QUERY):
                print("len(TRAINING_QUERIES[batch_keys[j]][positives]:"+str(len(TRAINING_QUERIES[batch_keys[j]]["positives"])))
                print("cfg.TRAIN_POSITIVES_PER_QUERY:"+str(cfg.TRAIN_POSITIVES_PER_QUERY))
                assert(0)
                faulty_tuple = True
                break

            # no cached feature vectors
            if (len(TRAINING_LATENT_VECTORS) == 0):
                q_tuples.append(
                    get_query_tuple_RGB_real(TRAINING_QUERIES[batch_keys[j]], cfg.TRAIN_POSITIVES_PER_QUERY, cfg.TRAIN_NEGATIVES_PER_QUERY,
                                    DB_QUERIES, hard_neg=[], other_neg=True))
                #print("q_tuples:"+str(q_tuples))
                # q_tuples.append(get_rotated_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_neg=[], other_neg=True))
                # q_tuples.append(get_jittered_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_neg=[], other_neg=True))

            elif (len(HARD_NEGATIVES.keys()) == 0):
                query = get_feature_representation(
                    TRAINING_QUERIES[batch_keys[j]]['query'], model)
                random.shuffle(TRAINING_QUERIES[batch_keys[j]]['negatives'])
                negatives = TRAINING_QUERIES[batch_keys[j]
                                             ]['negatives'][0:sampled_neg]
                hard_negs = get_random_hard_negatives(
                    query, negatives, num_to_take)
                q_tuples.append(
                    get_query_tuple_RGB_real(TRAINING_QUERIES[batch_keys[j]], cfg.TRAIN_POSITIVES_PER_QUERY, cfg.TRAIN_NEGATIVES_PER_QUERY,
                                    DB_QUERIES, hard_negs, other_neg=True))
                # q_tuples.append(get_rotated_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_negs, other_neg=True))
                # q_tuples.append(get_jittered_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_negs, other_neg=True))
            else:
                query = get_feature_representation(
                    TRAINING_QUERIES[batch_keys[j]]['query'], model)
                random.shuffle(TRAINING_QUERIES[batch_keys[j]]['negatives'])
                negatives = TRAINING_QUERIES[batch_keys[j]
                                             ]['negatives'][0:sampled_neg]
                hard_negs = get_random_hard_negatives(
                    query, negatives, num_to_take)
                hard_negs = list(set().union(
                    HARD_NEGATIVES[batch_keys[j]], hard_negs))
                q_tuples.append(
                    get_query_tuple_RGB_real(TRAINING_QUERIES[batch_keys[j]], cfg.TRAIN_POSITIVES_PER_QUERY, cfg.TRAIN_NEGATIVES_PER_QUERY,
                                    DB_QUERIES, hard_negs, other_neg=True))
                # q_tuples.append(get_rotated_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_negs, other_neg=True))
                # q_tuples.append(get_jittered_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_negs, other_neg=True))
            
            if (q_tuples[j][3].shape[2] != 3):
                no_other_neg = True
                break

        if(faulty_tuple):
            log_string('----' + str(i) + '-----')
            log_string('----' + 'FAULTY TUPLE' + '-----')
            continue

        if(no_other_neg):
            log_string('----' + str(i) + '-----')
            log_string('----' + 'NO OTHER NEG' + '-----')
            continue
        
        queries = []
        positives = []
        negatives = []
        other_neg = []
        for k in range(len(q_tuples)):
            queries.append(q_tuples[k][0])
            positives.append(q_tuples[k][1])
            negatives.append(q_tuples[k][2])
            other_neg.append(q_tuples[k][3])

        queries = np.array(queries, dtype=np.float32)
        queries = np.expand_dims(queries, axis=1)
        other_neg = np.array(other_neg, dtype=np.float32)
        other_neg = np.expand_dims(other_neg, axis=1)
        positives = np.array(positives, dtype=np.float32)
        negatives = np.array(negatives, dtype=np.float32)

        log_string('----' + str(i) + '-----')
        if (len(queries.shape) != 5):
            log_string('----' + 'FAULTY QUERY' + '-----')
            continue

        model.train()
        optimizer.zero_grad()
        
        output_queries, output_positives, output_negatives, output_other_neg = run_model(
            model, queries, positives, negatives, other_neg)
        
        #print("other_neg:"+str(other_neg.shape))
        loss = loss_function(output_queries, output_positives, output_negatives,  0.1,  use_min=cfg.TRIPLET_USE_BEST_POSITIVES, lazy=cfg.LOSS_LAZY, ignore_zero_loss=cfg.LOSS_IGNORE_ZERO_BATCH)
        #loss = loss_function(output_queries, output_positives, output_negatives,  rot_output_queries, rot_output_positives, rot_output_negatives, 0.1,  use_min=cfg.TRIPLET_USE_BEST_POSITIVES, lazy=False, ignore_zero_loss=cfg.LOSS_IGNORE_ZERO_BATCH)
        loss.backward()
        optimizer.step()

        log_string('batch loss: %f' % loss)
        train_writer.add_scalar("Loss", loss.cpu().item(), TOTAL_ITERATIONS)
        TOTAL_ITERATIONS += cfg.BATCH_NUM_QUERIES

        # EVALLLL
        '''
        if (epoch > 5 and i % (1400 // cfg.BATCH_NUM_QUERIES) == 29):
            TRAINING_LATENT_VECTORS = get_latent_vectors(
                model, TRAINING_QUERIES)
            print("Updated cached feature vectors")

        if (i % (6000 // cfg.BATCH_NUM_QUERIES) == 101):
            if isinstance(model, nn.DataParallel):
                model_to_save = model.module
            else:
                model_to_save = model
            save_name = cfg.LOG_DIR + cfg.MODEL_FILENAME
            torch.save({
                'epoch': epoch,
                'iter': TOTAL_ITERATIONS,
                'state_dict': model_to_save.state_dict(),
                'optimizer': optimizer.state_dict(),
            },
                save_name)
            print("Model Saved As " + save_name)
        '''

def get_feature_representation(filename, model):
    model.eval()
    queries = load_image_files([filename],False)
    queries = np.expand_dims(queries, axis=1)
    # if(BATCH_NUM_QUERIES-1>0):
    #    fake_queries=np.zeros((BATCH_NUM_QUERIES-1,1,NUM_POINTS,3))
    #    q=np.vstack((queries,fake_queries))
    # else:
    #    q=queries
    with torch.no_grad():
        q = torch.from_numpy(queries).float()
        q = q.to(device)
        output = model(q)
    output = output.detach().cpu().numpy()
    output = np.squeeze(output)
    model.train()
    return output


def get_random_hard_negatives(query_vec, random_negs, num_to_take):
    global TRAINING_LATENT_VECTORS

    latent_vecs = []
    for j in range(len(random_negs)):
        latent_vecs.append(TRAINING_LATENT_VECTORS[random_negs[j]])

    latent_vecs = np.array(latent_vecs)
    nbrs = KDTree(latent_vecs)
    distances, indices = nbrs.query(np.array([query_vec]), k=num_to_take)
    hard_negs = np.squeeze(np.array(random_negs)[indices[0]])
    hard_negs = hard_negs.tolist()
    return hard_negs


def get_latent_vectors(model, dict_to_process):
    train_file_idxs = np.arange(0, len(dict_to_process.keys()))

    batch_num = cfg.BATCH_NUM_QUERIES * \
        (1 + cfg.TRAIN_POSITIVES_PER_QUERY + cfg.TRAIN_NEGATIVES_PER_QUERY + 1)
    q_output = []

    model.eval()

    for q_index in range(len(train_file_idxs)//batch_num):
        file_indices = train_file_idxs[q_index *
                                       batch_num:(q_index+1)*(batch_num)]
        file_names = []
        for index in file_indices:
            file_names.append(dict_to_process[index]["query"])
        queries = load_image_files(file_names,False)

        feed_tensor = torch.from_numpy(queries).float()
        feed_tensor = feed_tensor.unsqueeze(1)
        feed_tensor = feed_tensor.to(device)
        with torch.no_grad():
            out = model(feed_tensor)

        out = out.detach().cpu().numpy()
        out = np.squeeze(out)

        q_output.append(out)

    q_output = np.array(q_output)
    if(len(q_output) != 0):
        q_output = q_output.reshape(-1, q_output.shape[-1])

    # handle edge case
    for q_index in range((len(train_file_idxs) // batch_num * batch_num), len(dict_to_process.keys())):
        index = train_file_idxs[q_index]
        queries = load_image_files([dict_to_process[index]["query"]],False)
        queries = np.expand_dims(queries, axis=1)

        # if (BATCH_NUM_QUERIES - 1 > 0):
        #    fake_queries = np.zeros((BATCH_NUM_QUERIES - 1, 1, NUM_POINTS, 3))
        #    q = np.vstack((queries, fake_queries))
        # else:
        #    q = queries

        #fake_pos = np.zeros((BATCH_NUM_QUERIES, POSITIVES_PER_QUERY, NUM_POINTS, 3))
        #fake_neg = np.zeros((BATCH_NUM_QUERIES, NEGATIVES_PER_QUERY, NUM_POINTS, 3))
        #fake_other_neg = np.zeros((BATCH_NUM_QUERIES, 1, NUM_POINTS, 3))
        #o1, o2, o3, o4 = run_model(model, q, fake_pos, fake_neg, fake_other_neg)
        with torch.no_grad():
            queries_tensor = torch.from_numpy(queries).float()
            o1 = model(queries_tensor.to(device))

        output = o1.detach().cpu().numpy()
        output = np.squeeze(output)
        if (q_output.shape[0] != 0):
            q_output = np.vstack((q_output, output))
        else:
            q_output = output

    model.train()
    return q_output


def run_model(model, queries, positives, negatives, other_neg, require_grad=True):
    queries_tensor = torch.from_numpy(queries).float()
    positives_tensor = torch.from_numpy(positives).float()
    negatives_tensor = torch.from_numpy(negatives).float()
    other_neg_tensor = torch.from_numpy(other_neg).float()
    feed_tensor = torch.cat(
        (queries_tensor, positives_tensor, negatives_tensor, other_neg_tensor), 1)
    feed_tensor = feed_tensor.view((-1, 1, cfg.SIZED_GRID_X, cfg.SIZED_GRID_Y, 3))
    
    feed_tensor.requires_grad_(require_grad)
    feed_tensor = feed_tensor.to(device)
    #print("feed_tensor:"+str(feed_tensor.shape))
    if require_grad:
        output = model(feed_tensor)
    else:
        with torch.no_grad():
            output = model(feed_tensor)
    # print("output:"+str(output.shape))

    output = output.view(cfg.BATCH_NUM_QUERIES, -1, output.shape[-1])
    #rot_output = rot_output.view(cfg.BATCH_NUM_QUERIES, -1, rot_output.shape[-1])
    o1, o2, o3, o4 = torch.split(
        output, [1, cfg.TRAIN_POSITIVES_PER_QUERY, cfg.TRAIN_NEGATIVES_PER_QUERY, 1], dim=1)
    #ro1, ro2, ro3, ro4 = torch.split(
    #    rot_output, [1, cfg.TRAIN_POSITIVES_PER_QUERY, cfg.TRAIN_NEGATIVES_PER_QUERY, 1], dim=1)
    return o1, o2, o3, o4#, ro1, ro2, ro3, ro4 

def get_query_tuple_RGB_real_ours(dict_value, num_pos, num_neg, QUERY_DICT, hard_neg=[], other_neg=False):
        # get query tuple for dictionary entry
        # return list [query,positives,negatives]
    query = load_image_file(dict_value["query"], full_path=False)  # Nx3
    #cv2.imwrite('/home/cc/Supervised-PointNetVlad_RGB/results/query.jpg', query)
    random.shuffle(dict_value["positives"])
    pos_files = []
    
    #print("dict_value[positives]:"+str(dict_value["positives"]))
    for i in range(num_pos):
        pos_files.append(QUERY_DICT[dict_value["positives"][i]]["query"])
    
    #print("pos_files:"+str(pos_files))
    positives = load_pos_neg_image_files(pos_files,full_path=False)
    '''
    cv2.imwrite('/home/cc/Supervised-PointNetVlad_RGB/results/color_img1.jpg', positives[0])
    cv2.imwrite('/home/cc/Supervised-PointNetVlad_RGB/results/color_img2.jpg', positives[1])
    '''
    neg_files = []
    neg_indices = []
    if(len(hard_neg) == 0):
        random.shuffle(dict_value["negatives"])
        #print("dict_value[negatives]:"+str(dict_value["negatives"]))
        for i in range(num_neg):
            neg_files.append(QUERY_DICT[dict_value["negatives"][i]]["query"])
            neg_indices.append(dict_value["negatives"][i])

    else:
        random.shuffle(dict_value["negatives"])
        for i in hard_neg:
            neg_files.append(QUERY_DICT[i]["query"])
            neg_indices.append(i)
        j = 0
        while(len(neg_files) < num_neg):
            if not dict_value["negatives"][j] in hard_neg:
                neg_files.append(
                    QUERY_DICT[dict_value["negatives"][j]]["query"])
                neg_indices.append(dict_value["negatives"][j])
            j += 1
    
    negatives = load_image_files(neg_files,full_path=False)
    '''
    cv2.imwrite('/home/cc/Supervised-PointNetVlad_RGB/results/neg_img1.jpg', negatives[0])
    cv2.imwrite('/home/cc/Supervised-PointNetVlad_RGB/results/neg_img2.jpg', negatives[1])
    cv2.imwrite('/home/cc/Supervised-PointNetVlad_RGB/results/neg_img3.jpg', negatives[2])
    cv2.imwrite('/home/cc/Supervised-PointNetVlad_RGB/results/neg_img4.jpg', negatives[3])
    cv2.imwrite('/home/cc/Supervised-PointNetVlad_RGB/results/neg_img5.jpg', negatives[4])
    assert(0)
    '''
    if other_neg is False:
        return [query, positives, negatives]
    # For Quadruplet Loss
    else:
        # get neighbors of negatives and query
        neighbors = []
        for pos in dict_value["positives"]:
            neighbors.append(pos)
        for neg in neg_indices:
            for pos in QUERY_DICT[neg]["positives"]:
                neighbors.append(pos)
        possible_negs = list(set(QUERY_DICT.keys())-set(neighbors))
        random.shuffle(possible_negs)

        if(len(possible_negs) == 0):
            return [query, positives, negatives, np.array([])]

        neg2 = load_image_file(QUERY_DICT[possible_negs[0]]["query"],full_path=False)
        return [query, positives, negatives, neg2]


if __name__ == "__main__":
    for i in range(1):
        train(i)