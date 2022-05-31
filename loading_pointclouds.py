import os
import pickle5 as pickle
import numpy as np
import random
import config as cfg
from open3d import read_point_cloud
import cv2
import random

def rotate_image(image, save_image = False):
    dim_1 = image.shape[1]
    cut_index = random.randint(0, dim_1-1)
    list_range = list(range(dim_1))
    new_list = list_range[cut_index:] + list_range[:cut_index]
    if save_image:
        print("here:"+str(os.path.join('./results/visualization_2/',"before_rotate.png")))
        cv2.imwrite(os.path.join('./results/visualization_2/',"before_rotate.png"), image)
    image =  image[:, new_list, :]
    if save_image:
        cv2.imwrite(os.path.join('./results/visualization_2/',"after_rotate.png"), image)
    
    return image

def get_queries_dict(filename):
    # key:{'query':file,'positives':[files],'negatives:[files], 'neighbors':[keys]}
    with open(filename, 'rb') as handle:
        print("filename:"+str(filename))
        queries = pickle.load(handle)
        print("Queries Loaded.")
        return queries


def get_sets_dict(filename):
    #[key_dataset:{key_pointcloud:{'query':file,'northing':value,'easting':value}},key_dataset:{key_pointcloud:{'query':file,'northing':value,'easting':value}}, ...}
    with open(filename, 'rb') as handle:
        trajectories = pickle.load(handle)
        print("Trajectories Loaded.")
        return trajectories


def load_pc_file(filename,full_path=False):
    # returns Nx3 matrix
    #print("filename:"+str(filename))
    if full_path:
        pc = read_point_cloud(os.path.join(filename))
    else:
        pc = read_point_cloud(os.path.join("/mnt/ab0fe826-9b3c-455c-bb72-5999d52034e0/deepmapping/benchmark_datasets/", filename))
    pc = np.asarray(pc.points, dtype=np.float32)
    
    if(pc.shape[0] != 256):
        print("Error in pointcloud shape")
        return np.array([])

    #pc = np.reshape(pc,(pc.shape[0]//3, 3))
    return pc


def load_pc_files(filenames,full_path):
    pcs = []
    for filename in filenames:
        # print(filename)
        pc = load_pc_file(filename,full_path=full_path)
        if(pc.shape[0] != 256):
            continue
        pcs.append(pc)
    pcs = np.array(pcs)
    return pcs

def load_image_file(filename, full_path=False):
    if full_path:
        image = cv2.imread(filename)
        dim = (cfg.SIZED_GRID_X,cfg.SIZED_GRID_Y)
        image = cv2.resize(image, dim,interpolation = cv2.INTER_AREA)
    else:
        image = cv2.imread(os.path.join("/mnt/NAS/home/cc/data/habitat_4/train/", filename))
        dim = (cfg.SIZED_GRID_X,cfg.SIZED_GRID_Y)
        image = cv2.resize(image, dim,interpolation = cv2.INTER_AREA)
        # print("image2:"+str(image.shape))
        # assert(0)
    image = np.asarray(image, dtype=np.float32)

    if(image.shape[2] != 3):
        print("Error in Image shape")
        return np.array([])
    #pc = np.reshape(pc,(pc.shape[0]//3, 3))
    return image


def load_image_files(filenames,full_path):
    images = []
    for filename in filenames:
        image = load_image_file(filename, full_path=full_path)
        images.append(image)
    images = np.asarray(images, dtype=np.float32)
    return images

def load_pos_neg_image_files(filenames,full_path):
    images = []
    for filename in filenames:
        image = load_image_file(filename, full_path=full_path)
        for i in range(2):
            images.append(rotate_image(image,False))
    images = np.asarray(images, dtype=np.float32)
    return images

def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        #-90 to 90
        rotation_angle = (np.random.uniform()*np.pi) - np.pi/2.0
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, -sinval, 0],
                                    [sinval, cosval, 0],
                                    [0, 0, 1]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(
            shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.005, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data


def get_query_tuple(dict_value, num_pos, num_neg, QUERY_DICT, hard_neg=[], other_neg=False):
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


def get_rotated_tuple(dict_value, num_pos, num_neg, QUERY_DICT, hard_neg=[], other_neg=False):
    query = load_pc_file(dict_value["query"])  # Nx3
    q_rot = rotate_point_cloud(np.expand_dims(query, axis=0))
    q_rot = np.squeeze(q_rot)

    random.shuffle(dict_value["positives"])
    pos_files = []
    for i in range(num_pos):
        pos_files.append(QUERY_DICT[dict_value["positives"][i]]["query"])
    #positives= load_pc_files(dict_value["positives"][0:num_pos])
    positives = load_pc_files(pos_files)
    p_rot = rotate_point_cloud(positives)

    neg_files = []
    neg_indices = []
    if(len(hard_neg) == 0):
        random.shuffle(dict_value["negatives"])
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
    negatives = load_pc_files(neg_files)
    n_rot = rotate_point_cloud(negatives)

    if other_neg is False:
        return [q_rot, p_rot, n_rot]

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
            return [q_jit, p_jit, n_jit, np.array([])]

        neg2 = load_pc_file(QUERY_DICT[possible_negs[0]]["query"])
        n2_rot = rotate_point_cloud(np.expand_dims(neg2, axis=0))
        n2_rot = np.squeeze(n2_rot)

        return [q_rot, p_rot, n_rot, n2_rot]


def get_jittered_tuple(dict_value, num_pos, num_neg, QUERY_DICT, hard_neg=[], other_neg=False):
    query = load_pc_file(dict_value["query"])  # Nx3
    #q_rot= rotate_point_cloud(np.expand_dims(query, axis=0))
    q_jit = jitter_point_cloud(np.expand_dims(query, axis=0))
    q_jit = np.squeeze(q_jit)

    random.shuffle(dict_value["positives"])
    pos_files = []
    for i in range(num_pos):
        pos_files.append(QUERY_DICT[dict_value["positives"][i]]["query"])
    #positives= load_pc_files(dict_value["positives"][0:num_pos])
    positives = load_pc_files(pos_files)
    p_jit = jitter_point_cloud(positives)

    neg_files = []
    neg_indices = []
    if(len(hard_neg) == 0):
        random.shuffle(dict_value["negatives"])
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
    negatives = load_pc_files(neg_files)
    n_jit = jitter_point_cloud(negatives)

    if other_neg is False:
        return [q_jit, p_jit, n_jit]

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
            return [q_jit, p_jit, n_jit, np.array([])]

        neg2 = load_pc_file(QUERY_DICT[possible_negs[0]]["query"])
        n2_jit = jitter_point_cloud(np.expand_dims(neg2, axis=0))
        n2_jit = np.squeeze(n2_jit)

        return [q_jit, p_jit, n_jit, n2_jit]
