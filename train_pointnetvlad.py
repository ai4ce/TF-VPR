import torch
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print("device:"+str(device))
import argparse
import importlib
import math
import os
import socket
import sys
sys.path.append('/usr/local/lib/python3.6/dist-packages/python_pcl-0.3-py3.6-linux-x86_64.egg/')
import pcl

import numpy as np
from sklearn.neighbors import KDTree, NearestNeighbors

import config as cfg
import evaluate
import loss.pointnetvlad_loss as PNV_loss
import models.PointNetVlad as PNV
import models.Verification as VFC
import generating_queries.generate_training_tuples_cc_baseline_batch as generate_dataset

import torch
import torch.nn as nn
from loading_pointclouds import *
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.backends import cudnn

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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
parser.add_argument('--max_epoch', type=int, default=20,
                    help='Epoch to run [default: 20]')
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
parser.add_argument('--loss_function', default='quadruplet', choices=[
                    'triplet', 'quadruplet'], help='triplet or quadruplet [default: quadruplet]')
parser.add_argument('--loss_not_lazy', action='store_false',
                    help='If present, do not use lazy variant of loss')
parser.add_argument('--loss_ignore_zero_batch', action='store_true',
                    help='If present, mean only batches with loss > 0.0')
parser.add_argument('--triplet_use_best_positives', action='store_true',
                    help='If present, use best positives, otherwise use hardest positives')
parser.add_argument('--resume', action='store_true',
                    help='If present, restore checkpoint and resume training')
parser.add_argument('--dataset_folder', default='../../dataset/',
                    help='PointNetVlad Dataset Folder')

FLAGS = parser.parse_args()
cfg.BATCH_NUM_QUERIES = FLAGS.batch_num_queries
#cfg.EVAL_BATCH_SIZE = 12
cfg.NUM_POINTS = 256
cfg.TRAIN_POSITIVES_PER_QUERY = FLAGS.positives_per_query
cfg.TRAIN_NEGATIVES_PER_QUERY = FLAGS.negatives_per_query
cfg.MAX_EPOCH = FLAGS.max_epoch
cfg.BASE_LEARNING_RATE = FLAGS.learning_rate
cfg.MOMENTUM = FLAGS.momentum
cfg.OPTIMIZER = FLAGS.optimizer
cfg.DECAY_STEP = FLAGS.decay_step
cfg.DECAY_RATE = FLAGS.decay_rate
cfg.MARGIN1 = FLAGS.margin_1
cfg.MARGIN2 = FLAGS.margin_2
cfg.FEATURE_OUTPUT_DIM = 256

cfg.LOSS_FUNCTION = FLAGS.loss_function
cfg.TRIPLET_USE_BEST_POSITIVES = FLAGS.triplet_use_best_positives
cfg.LOSS_LAZY = FLAGS.loss_not_lazy
cfg.LOSS_IGNORE_ZERO_BATCH = FLAGS.loss_ignore_zero_batch

cfg.TRAIN_FILE = 'generating_queries/training_queries_baseline.pickle'
cfg.TEST_FILE = 'generating_queries/test_queries_baseline.pickle'

cfg.LOG_DIR = FLAGS.log_dir
if not os.path.exists(cfg.LOG_DIR):
    os.mkdir(cfg.LOG_DIR)
LOG_FOUT = open(os.path.join(cfg.LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

cfg.RESULTS_FOLDER = FLAGS.results_dir
print("cfg.RESULTS_FOLDER:"+str(cfg.RESULTS_FOLDER))

# Load dictionary of training queries
#TRAINING_QUERIES = get_queries_dict(cfg.TRAIN_FILE)
#TEST_QUERIES = get_queries_dict(cfg.TEST_FILE)
#TRAINING_QUERIES_init = get_queries_dict(cfg.TRAIN_FILE)
#TEST_QUERIES_init = get_queries_dict(cfg.TEST_FILE)

cfg.BN_INIT_DECAY = 0.5
cfg.BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(cfg.DECAY_STEP)
cfg.BN_DECAY_CLIP = 0.99

HARD_NEGATIVES = {}
TRAINING_LATENT_VECTORS = []

TOTAL_ITERATIONS = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def train():
    global HARD_NEGATIVES, TOTAL_ITERATIONS
    #global TRAINING_QUERIES, TEST_QUERIES
    bn_decay = get_bn_decay(0)
    #tf.summary.scalar('bn_decay', bn_decay)

    #loss = lazy_quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg_vec, MARGIN1, MARGIN2)
    if cfg.LOSS_FUNCTION == 'quadruplet':
        loss_function = PNV_loss.quadruplet_loss
    else:
        loss_function = PNV_loss.triplet_loss_wrapper
    learning_rate = get_learning_rate(0)
    
    train_writer = SummaryWriter(os.path.join(cfg.LOG_DIR, 'train'))
    #test_writer = SummaryWriter(os.path.join(cfg.LOG_DIR, 'test'))

    model = PNV.PointNetVlad(global_feat=True, feature_transform=True,
                             max_pool=False, output_dim=cfg.FEATURE_OUTPUT_DIM, num_points=cfg.NUM_POINTS)
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
        TOTAL_ITERATIONS = starting_epoch * len(TRAINING_QUERIES)

        model.load_state_dict(saved_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        starting_epoch = 0

    #model = nn.DataParallel(model)

    LOG_FOUT.write(cfg.cfg_str())
    LOG_FOUT.write("\n")
    LOG_FOUT.flush()

    data_index = 0
    potential_positives = None
    potential_distributions = None
    trusted_positives = None

    all_folders = sorted(os.listdir(cfg.DATASET_FOLDER))
    
    folders = []
    # All runs are used for training (both full and partial)
    index_list = range(len(all_folders))
    for index in index_list:
        folders.append(all_folders[index])
    
    for folder in folders:
        all_files = list(sorted(os.listdir(os.path.join(cfg.DATASET_FOLDER,folder))))
        all_files.remove('gt_pose.mat')
        all_files.remove('gt_pose.png')

    for epoch in range(starting_epoch, cfg.MAX_EPOCH):
        print(epoch)
        print()
        print("trusted_positives:"+str(np.array(trusted_positives).shape))
        generate_dataset.generate(data_index, definite_positives=trusted_positives, inside=False)  
        TRAIN_FILE = 'generating_queries/train_pickle/training_queries_baseline_'+str(data_index)+'.pickle'
        TEST_FILE = 'generating_queries/train_pickle/test_queries_baseline_'+str(data_index)+'.pickle'
        data_index = data_index+1
        # Load dictionary of training queries
        TRAINING_QUERIES = get_queries_dict(TRAIN_FILE)
        TEST_QUERIES = get_queries_dict(TEST_FILE)

        log_string('**** EPOCH %03d ****' % (epoch))
        sys.stdout.flush()

        train_one_epoch(model, optimizer, train_writer, loss_function, epoch)
        
        log_string('EVALUATING...')
        cfg.OUTPUT_FILE = cfg.RESULTS_FOLDER + 'results_' + str(epoch) + '.txt'

        #eval_recall, db_vec = evaluate.evaluate_model(model, True) #db_vec gives the evaluate nearest neighbours, folder* 2048* positves_dim
        _, db_vec = evaluate.evaluate_model(model, epoch, True, full_pickle=True)
        eval_recall, _ = evaluate.evaluate_model(model, epoch, True, full_pickle=False)
        if potential_positives is None:
            potential_positives = []
            potential_distributions = []
            trusted_positives = []
            db_vec = np.array(db_vec)
            for index in range(db_vec.shape[0]):
                trusted_positive = []
                nbrs = NearestNeighbors(n_neighbors=cfg.EVAL_NEAREST, algorithm='ball_tree').fit(db_vec[index])
                distance, indice = nbrs.kneighbors(db_vec[index])
                #print("distance:"+str(np.exp(-distance*10)[0]))
                #print("indice:"+str(indice[0]))
                weight = np.exp(-distance*10).tolist()

                potential_positives.append(indice.tolist())
                potential_distributions.append(np.exp(-distance*10).tolist())
                #print("np.array(indice)[np.argsort(weight)[::-1][0]]:"+str(np.array(indice)[np.argsort(weight[:,::-1])].shape))
                #assert(0)
                for index2 in range(db_vec.shape[1]):
                    pre_trusted_positive = np.array(indice[index2])[np.argsort(weight[index2])[::-1][:(cfg.INIT_TRUST)]]
                    folder_path = os.path.join(cfg.DATASET_FOLDER,folders[index])
                    
                    #print("pre_trusted_positive:"+str(pre_trusted_positive))
                    trusted_pos = VFC.filter_trusted(folder_path, all_files, index2, pre_trusted_positive)  
                    #print("trusted_positive:"+str(trusted_positive))
                    trusted_positive.append(trusted_pos)

                trusted_positives.append(trusted_positive)
        else:
            new_potential_positives = []
            new_potential_distributions = []
            new_trusted_positives = []
            db_vec = np.array(db_vec)
            for index in range(db_vec.shape[0]):
                new_potential_positive = []
                new_potential_distribution = []
                new_trusted_positive = []
                nbrs = NearestNeighbors(n_neighbors=cfg.EVAL_NEAREST, algorithm='ball_tree').fit(db_vec[index])
                distance, indice = nbrs.kneighbors(db_vec[index])
                weight = np.exp(-distance*10).tolist()
                for index2 in range(db_vec.shape[1]):
                    pos_set = potential_positives[index][index2]
                    pos_dis = potential_distributions[index][index2]
                    for count,inc in enumerate(indice[index2]):
                        if inc not in pos_set:
                            pos_set.append(inc)
                            pos_dis.append(weight[index2][count])
                        else:
                            pos_dis[pos_set.index(inc)] = pos_dis[pos_set.index(inc)] + weight[index2][count] # if the element exists, distribution +1
                    #print("np.argsort:",np.argsort(pos_dis)[::-1][:5])
                    #print("pos_set[np.argsort(pos_dis)[:5]:]"+str(np.array(pos_set)[np.argsort(pos_dis)[::-1][:5]]))
                    #assert(0)
                    new_potential_positive.append(pos_set)
                    new_potential_distribution.append(pos_dis)
                    '''
                    if data_index-1 >=5:
                        data_index_constraint = 6
                    else:
                        data_index_constraint = data_index
                    '''
                    pre_trusted_positive = np.array(pos_set)[np.argsort(pos_dis)[::-1][:(cfg.INIT_TRUST+int(data_index-1)//cfg.INIT_TRUST_SCALAR)]]
                    #print("pre_trusted_positive:"+str(pre_trusted_positive.shape))
                    trusted_positive = VFC.filter_trusted(folder_path, all_files, index2, pre_trusted_positive)
                    #print("trusted_positive:"+str(trusted_positive))
                    new_trusted_positive.append(trusted_positive)

                new_potential_positives.append(new_potential_positive)
                new_potential_distributions.append(new_potential_distribution)
                new_trusted_positives.append(new_trusted_positive)
            potential_positives = new_potential_positives
            potential_distributions = new_potential_distributions
            trusted_positives = new_trusted_positives

            #print("potential_positives:"+str(potential_positives[0][1]))
            #print("potential_distributions:"+str(potential_distributions[0][1]))
        #print("trusted_positives:"+str(np.array(trusted_positives)[1][2]))
        log_string('EVAL RECALL: %s' % str(eval_recall))

        train_writer.add_scalar("Val Recall", eval_recall, epoch)


def train_one_epoch(model, optimizer, train_writer, loss_function, epoch):
    global HARD_NEGATIVES
    global TRAINING_LATENT_VECTORS, TOTAL_ITERATIONS

    is_training = True
    sampled_neg = 4000
    # number of hard negatives in the training tuple
    # which are taken from the sampled negatives
    num_to_take = 10

    # Shuffle train files
    #print("TRAINING_QUERIES:"+str(TRAINING_QUERIES))
    train_file_idxs = np.arange(0, len(TRAINING_QUERIES.keys()))
    np.random.shuffle(train_file_idxs)
    for i in range(len(train_file_idxs)//cfg.BATCH_NUM_QUERIES):
    #for i in range(1):
        # for i in range (5):
        batch_keys = train_file_idxs[i *
                                     cfg.BATCH_NUM_QUERIES:(i+1)*cfg.BATCH_NUM_QUERIES]
        q_tuples = []

        faulty_tuple = False
        no_other_neg = False
        for j in range(cfg.BATCH_NUM_QUERIES):
            if (len(TRAINING_QUERIES[batch_keys[j]]["positives"]) < cfg.TRAIN_POSITIVES_PER_QUERY):
                faulty_tuple = True
                break

            # no cached feature vectors
            if (len(TRAINING_LATENT_VECTORS) == 0):
                q_tuples.append(
                    get_query_tuple(TRAINING_QUERIES[batch_keys[j]], cfg.TRAIN_POSITIVES_PER_QUERY, cfg.TRAIN_NEGATIVES_PER_QUERY,
                                    TRAINING_QUERIES, hard_neg=[], other_neg=True))
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
                    get_query_tuple(TRAINING_QUERIES[batch_keys[j]], cfg.TRAIN_POSITIVES_PER_QUERY, cfg.TRAIN_NEGATIVES_PER_QUERY,
                                    TRAINING_QUERIES, hard_negs, other_neg=True))
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
                    get_query_tuple(TRAINING_QUERIES[batch_keys[j]], cfg.TRAIN_POSITIVES_PER_QUERY, cfg.TRAIN_NEGATIVES_PER_QUERY,
                                    TRAINING_QUERIES, hard_negs, other_neg=True))
                # q_tuples.append(get_rotated_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_negs, other_neg=True))
                # q_tuples.append(get_jittered_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_negs, other_neg=True))
            
            if (q_tuples[j][3].shape[0] != cfg.NUM_POINTS):
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
        if (len(queries.shape) != 4):
            log_string('----' + 'FAULTY QUERY' + '-----')
            continue

        model.train()
        optimizer.zero_grad()
        
        output_queries, output_positives, output_negatives, output_other_neg = run_model(
            model, queries, positives, negatives, other_neg)
        #print("output_queries:"+str(output_queries))
        #print("output_positives:"+str(output_positives))
        #print("output_negatives:"+str(output_negatives))
        #print("output_other_neg:"+str(output_other_neg))

        #print("negatives:"+str(negatives))
        #print("other_neg:"+str(other_neg))
        #print("queries:"+str(queries))

        loss = loss_function(output_queries, output_positives, output_negatives, output_other_neg, cfg.MARGIN_1, cfg.MARGIN_2, use_min=cfg.TRIPLET_USE_BEST_POSITIVES, lazy=cfg.LOSS_LAZY, ignore_zero_loss=cfg.LOSS_IGNORE_ZERO_BATCH)
        loss.backward()
        optimizer.step()

        log_string('batch loss: %f' % loss)
        train_writer.add_scalar("Loss", loss.cpu().item(), TOTAL_ITERATIONS)
        TOTAL_ITERATIONS += cfg.BATCH_NUM_QUERIES

        # EVALLLL
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


def get_feature_representation(filename, model):
    model.eval()
    queries = load_pc_files([filename],True)
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
        queries = load_pc_files(file_names,True)

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
        queries = load_pc_files([dict_to_process[index]["query"]],True)
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
            queries_tensor = torch.from_numpy(queries).float().cuda()
            o1 = model(queries_tensor)

        output = o1.detach().cpu().numpy()
        output = np.squeeze(output)
        if (q_output.shape[0] != 0):
            q_output = np.vstack((q_output, output))
        else:
            q_output = output

    model.train()
    print(q_output.shape)
    return q_output


def run_model(model, queries, positives, negatives, other_neg, require_grad=True):
    queries_tensor = torch.from_numpy(queries).float()
    positives_tensor = torch.from_numpy(positives).float()
    negatives_tensor = torch.from_numpy(negatives).float()
    other_neg_tensor = torch.from_numpy(other_neg).float()
    feed_tensor = torch.cat(
        (queries_tensor, positives_tensor, negatives_tensor, other_neg_tensor), 1)
    feed_tensor = feed_tensor.view((-1, 1, cfg.NUM_POINTS, 3))
    feed_tensor.requires_grad_(require_grad)
    feed_tensor = feed_tensor.to(device)
    #print("feed_tensor:"+str(feed_tensor))
    if require_grad:
        output = model(feed_tensor)
    else:
        with torch.no_grad():
            output = model(feed_tensor)
    #print("output:"+str(output))
    output = output.view(cfg.BATCH_NUM_QUERIES, -1, cfg.FEATURE_OUTPUT_DIM)
    o1, o2, o3, o4 = torch.split(
        output, [1, cfg.TRAIN_POSITIVES_PER_QUERY, cfg.TRAIN_NEGATIVES_PER_QUERY, 1], dim=1)

    return o1, o2, o3, o4


if __name__ == "__main__":
    train()
