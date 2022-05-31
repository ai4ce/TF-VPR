import argparse
import math
import numpy as np
import socket
import importlib
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.backends import cudnn

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from loading_pointclouds import *
import models.PointNetVlad as PNV
from tensorboardX import SummaryWriter
import loss.pointnetvlad_loss

import config as cfg

cudnn.enabled = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate():
    model = PNV.PointNetVlad(global_feat=True, feature_transform=True, max_pool=False,
                                      output_dim=cfg.FEATURE_OUTPUT_DIM, num_points=cfg.NUM_POINTS)
    model = model.to(device)

    resume_filename = cfg.LOG_DIR + "checkpoint.pth.tar"
    print("Resuming From ", resume_filename)
    checkpoint = torch.load(resume_filename)
    saved_state_dict = checkpoint['state_dict']
    model.load_state_dict(saved_state_dict)

    #model = nn.DataParallel(model)
    ave_one_percent_recall = evaluate_model(model)
    print("ave_one_percent_recall:"+str(ave_one_percent_recall))


def evaluate_model(model,optimizer,epoch,save=False):
    if save:
        torch.save({
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            }, cfg.LOG_DIR + "checkpoint.pth.tar")
    
    #checkpoint = torch.load(cfg.LOG_DIR + "checkpoint.pth.tar")
    #saved_state_dict = checkpoint['state_dict']
    #model.load_state_dict(saved_state_dict)
    DATABASE_SETS = get_sets_dict(cfg.EVAL_DATABASE_FILE)

    QUERY_SETS = get_sets_dict(cfg.EVAL_QUERY_FILE)
    '''
    QUERY_SETS = []
    for i in range(4):
        QUERY = {}
        for j in range(len(QUERY_SETS_temp)//4):
            #QUERY[len(QUERY.keys())] = {"query":QUERY_SETS_temp[i][j]['query'],
            #                                "x":float(QUERY_SETS_temp[i][j]['x']),
            #                                "y":float(QUERY_SETS_temp[i][j]['y']),
            #                                }
            QUERY[len(QUERY.keys())] = QUERY_SETS_temp[i][j]
        QUERY_SETS.append(QUERY)
    '''
    if not os.path.exists(cfg.RESULTS_FOLDER):
        os.mkdir(cfg.RESULTS_FOLDER)

    recall_1 = np.zeros(20)
    recall_5 = np.zeros(102)
    recall_10 = np.zeros(205)
    count = 0

    similarity_1 = []
    similarity_5 = []
    similarity_10 = []

    one_percent_recall = []
    five_percent_recall = []
    ten_percent_recall = []

    DATABASE_VECTORS = []
    QUERY_VECTORS = []

    for i in range(len(DATABASE_SETS)):
        DATABASE_VECTORS.append(get_latent_vectors(model, DATABASE_SETS[i]))
    
    for j in range(len(QUERY_SETS)):
        QUERY_VECTORS.append(get_latent_vectors(model, QUERY_SETS[j]))

    len_tr = np.array(DATABASE_VECTORS).shape[1]
    recall_1 = np.zeros(int(round(len_tr/100)))
    recall_5 = np.zeros(int(round(len_tr/20)))
    recall_10 = np.zeros(int(round(len_tr/10)))
    #############
    for m in range(len(QUERY_SETS)):
        for n in range(len(QUERY_SETS)):
            if (m == n):
                continue
            pair_recall_1, pair_recall_5, pair_recall_10, pair_similarity_1, pair_similarity_5, pair_similarity_10, pair_opr_1, pair_opr_5, pair_opr_10 = get_recall(
                m, n, DATABASE_VECTORS, QUERY_VECTORS, QUERY_SETS)
            recall_1 += np.array(pair_recall_1)
            recall_5 += np.array(pair_recall_5)
            recall_10 += np.array(pair_recall_10)

            count += 1
            one_percent_recall.append(pair_opr_1)
            five_percent_recall.append(pair_opr_5)
            ten_percent_recall.append(pair_opr_10)

            for x in pair_similarity_1:
                similarity_1.append(x)
            for x in pair_similarity_5:
                similarity_5.append(x)
            for x in pair_similarity_10:
                similarity_10.append(x)
    #########
    
    
    ### Save Evaluate vectors
    file_name = os.path.join(cfg.RESULTS_FOLDER, "database"+str(epoch)+".npy")
    np.save(file_name, np.array(DATABASE_VECTORS))
    print("saving for DATABASE_VECTORS to "+str(file_name))
    
    ave_recall_1 = recall_1 / count
    ave_recall_5 = recall_5 / count
    ave_recall_10 = recall_10 / count
    # print(ave_recall)

    # print(similarity)
    average_similarity_1 = np.mean(similarity_1)
    average_similarity_5 = np.mean(similarity_5)
    average_similarity_10 = np.mean(similarity_10)
    # print(average_similarity)

    ave_one_percent_recall = np.mean(one_percent_recall)
    ave_five_percent_recall = np.mean(five_percent_recall)
    ave_ten_percent_recall = np.mean(ten_percent_recall)
    # print(ave_one_percent_recall)
    
    #print("os.path.join(/home/cc/PointNet-torch2,cfg.OUTPUT_FILE,log.txt):"+str(os.path.join("/home/cc/PointNet-torch2",cfg.OUTPUT_FILE,"log.txt")))
    #assert(0)
    with open(os.path.join(cfg.OUTPUT_FILE), "w") as output:
        output.write("Average Recall @1:\n")
        output.write(str(ave_recall_1)+"\n")
        output.write("Average Recall @5:\n")
        output.write(str(ave_recall_5)+"\n")
        output.write("Average Recall @10:\n")
        output.write(str(ave_recall_10)+"\n")
        output.write("\n\n")
        output.write("Average Similarity_1:\n")
        output.write(str(average_similarity_1)+"\n")
        output.write("Average Similarity_5:\n")
        output.write(str(average_similarity_5)+"\n")
        output.write("Average Similarity_10:\n")
        output.write(str(average_similarity_10)+"\n")
        output.write("\n\n")
        output.write("Average Top 1% Recall:\n")
        output.write(str(ave_one_percent_recall)+"\n")
        output.write("Average Top 5% Recall:\n")
        output.write(str(ave_five_percent_recall)+"\n")
        output.write("Average Top 10% Recall:\n")
        output.write(str(ave_ten_percent_recall)+"\n")
    
    return ave_one_percent_recall, ave_five_percent_recall, ave_ten_percent_recall


def get_latent_vectors(model, dict_to_process):
    model.eval()
    is_training = False
    train_file_idxs = np.arange(0, len(dict_to_process.keys()))

    batch_num = cfg.EVAL_BATCH_SIZE * \
        (1 + cfg.EVAL_POSITIVES_PER_QUERY + cfg.EVAL_NEGATIVES_PER_QUERY)
    q_output = []
    for q_index in range(len(train_file_idxs)//batch_num):
        file_indices = train_file_idxs[q_index *
                                       batch_num:(q_index+1)*(batch_num)]
        file_names = []
        for index in file_indices:
            file_names.append(dict_to_process[index]["query"])
        queries = load_pc_files(file_names,True)

        with torch.no_grad():
            feed_tensor = torch.from_numpy(queries).float()
            feed_tensor = feed_tensor.unsqueeze(1)
            feed_tensor = feed_tensor.to(device)
            out = model(feed_tensor)

        out = out.detach().cpu().numpy()
        out = np.squeeze(out)

        #out = np.vstack((o1, o2, o3, o4))
        q_output.append(out)

    q_output = np.array(q_output)
    if(len(q_output) != 0):
        q_output = q_output.reshape(-1, q_output.shape[-1])

    # handle edge case
    index_edge = len(train_file_idxs) // batch_num * batch_num
    if index_edge < len(dict_to_process.keys()):
        file_indices = train_file_idxs[index_edge:len(dict_to_process.keys())]
        file_names = []

        for index in file_indices:
            file_names.append(dict_to_process[index]["query"])
        queries = load_pc_files(file_names,True)

        with torch.no_grad():
            feed_tensor = torch.from_numpy(queries).float()
            feed_tensor = feed_tensor.unsqueeze(1)
            feed_tensor = feed_tensor.to(device)
            o1 = model(feed_tensor)

        output = o1.detach().cpu().numpy()
        output = np.squeeze(output)
        if (q_output.shape[0] != 0):
            q_output = np.vstack((q_output, output))
        else:
            q_output = output

    model.train()

    return q_output


def get_recall(m, n, DATABASE_VECTORS, QUERY_VECTORS, QUERY_SETS):
    database_output = DATABASE_VECTORS[m]  #2048*256
    queries_output = QUERY_VECTORS[n]      #10*256
    
    database_nbrs = KDTree(database_output)
    num_neighbors = 25

    recalls = []
    similarity_scores = []
    N_percent_recalls = []

    percent_array = [100, 20, 10]
    for percent in percent_array:
        threshold = max(int(round(len(database_output)/percent)), 1)
        recall_N = [0] * threshold
        topN_similarity_score = []
        N_percent_retrieved = 0

        num_evaluated = 0
        for i in range(len(queries_output)):
            true_neighbors = QUERY_SETS[n][i][m]
            if(len(true_neighbors) == 0):
                continue
            num_evaluated += 1
            distances, indices = database_nbrs.query(
                np.array([queries_output[i]]),k=threshold)
            
            #indices = indices + n*2048
            for j in range(len(indices[0])):
                if indices[0][j] in true_neighbors:
                    if(j == 0):
                        similarity = np.dot(
                            queries_output[i], database_output[indices[0][j]])
                        topN_similarity_score.append(similarity)
                    recall_N[j] += 1
                    break

            if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
                N_percent_retrieved += 1
        
        if float(num_evaluated)!=0:
            N_percent_recall = (N_percent_retrieved/float(num_evaluated))*100
            recall_N = (np.cumsum(recall_N)/float(num_evaluated))*100
        else:
            N_percent_recall = 0
            recall_N = 0
        recalls.append(recall_N)
        similarity_scores.append(topN_similarity_score)
        N_percent_recalls.append(N_percent_recall)

    recall_1, recall_5, recall_10 = recalls[0], recalls[1], recalls[2] 
    top1_similarity_score, top5_similarity_score, top10_similarity_score = similarity_scores[0], similarity_scores[1], similarity_scores[2] 
    one_percent_recall, five_percent_recall, ten_percent_recall = N_percent_recalls[0], N_percent_recalls[1], N_percent_recalls[2] 

    return recall_1, recall_5, recall_10, top1_similarity_score, top5_similarity_score, top10_similarity_score, one_percent_recall, five_percent_recall, ten_percent_recall


if __name__ == "__main__":
    # params
    parser = argparse.ArgumentParser()
    parser.add_argument('--positives_per_query', type=int, default=4,
                        help='Number of potential positives in each training tuple [default: 2]')
    parser.add_argument('--negatives_per_query', type=int, default=12,
                        help='Number of definite negatives in each training tuple [default: 20]')
    parser.add_argument('--eval_batch_size', type=int, default=12,
                        help='Batch Size during training [default: 1]')
    parser.add_argument('--dimension', type=int, default=256)
    parser.add_argument('--decay_step', type=int, default=200000,
                        help='Decay step for lr decay [default: 200000]')
    parser.add_argument('--decay_rate', type=float, default=0.7,
                        help='Decay rate for lr decay [default: 0.8]')
    parser.add_argument('--results_dir', default='results/',
                        help='results dir [default: results]')
    parser.add_argument('--dataset_folder', default='../../dataset/',
                        help='PointNetVlad Dataset Folder')
    FLAGS = parser.parse_args()

    #BATCH_SIZE = FLAGS.batch_size
    #cfg.EVAL_BATCH_SIZE = FLAGS.eval_batch_size
    cfg.NUM_POINTS = 4096
    cfg.FEATURE_OUTPUT_DIM = 256
    cfg.EVAL_POSITIVES_PER_QUERY = FLAGS.positives_per_query
    cfg.EVAL_NEGATIVES_PER_QUERY = FLAGS.negatives_per_query
    cfg.DECAY_STEP = FLAGS.decay_step
    cfg.DECAY_RATE = FLAGS.decay_rate

    cfg.RESULTS_FOLDER = FLAGS.results_dir

    cfg.EVAL_DATABASE_FILE = 'generating_queries/evaluation_database.pickle'
    cfg.EVAL_QUERY_FILE = 'generating_queries/evaluation_query.pickle'

    cfg.LOG_DIR = 'log/'
    cfg.OUTPUT_FILE = cfg.RESULTS_FOLDER + 'results.txt'
    cfg.MODEL_FILENAME = "model.ckpt"

    cfg.DATASET_FOLDER = FLAGS.dataset_folder

    evaluate()
