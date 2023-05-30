
import random
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler,StandardScaler

def fixed_opt(fixed_len, vectors_list, seq_len_list):
    vec_mat = []

    for i in range(len(vectors_list)):
        # print(i)
        temp_arr = np.zeros((fixed_len, len(vectors_list[i][0])))
        seq_len = len(vectors_list[i])
        if seq_len > fixed_len:
            seq_len_list[i] = fixed_len
        temp_len = min(seq_len, fixed_len)
        temp_arr[:temp_len, :] = vectors_list[i][:temp_len, :]
        vec_mat.append(temp_arr)

    return np.array(vec_mat), seq_len_list

def read_base_mat4res(in_file, fixed_len):
    # 从BLM/BioSeq-BLM/results/res_features.txt 中读取特征
    vectors_list = []
    seq_len_list = []
    f = open(in_file, 'r')
    lines = f.readlines()
    vectors = []
    flag = 0
    for line in lines:
        if len(line.strip()) != 0:
            if line[0] != '>':
                vector = line.strip().split('\t')
                vector = list(map(float, vector))
                vectors.append(vector)
                flag = 1
            else:
                if flag == 1:
                    seq_len_list.append(len(vectors))
                    vectors_list.append(np.array(vectors))
                    vectors = []
                    flag = 0
    f.close()
    vec_mat, fixed_seq_len_list = fixed_opt(fixed_len, vectors_list, seq_len_list)
    return vec_mat, fixed_seq_len_list

#读取label
def res_dl_label_read(res_label_list, fixed_len):
    res_label_mat = []
    for res_label in res_label_list:
        temp_arr = np.zeros(fixed_len)
        seq_len = len(res_label)
        temp_len = min(seq_len, fixed_len)
        temp_arr[:temp_len] = res_label[:temp_len]
        res_label_mat.append(temp_arr)

    return np.array(res_label_mat)

#  生成label矩阵
def read_res_label_file(label_file):
    
    res_labels_list = []
    label_len_list = []

    f = open(label_file, 'r')
    lines = f.readlines()
    for line in lines:
        if line[0] != '>':
            labels = line.strip().split()
            labels = list(map(int, labels))
            label_len_list.append(len(labels))
            res_labels_list.append(labels)
    f.close()

    return res_labels_list, label_len_list

def getFeaturesLabels(fea_file, label_file, fix_len):
    res_labels_list, label_len_list = read_res_label_file(label_file)
    vec_mat, fixed_seq_len_list = read_base_mat4res(fea_file, fix_len)
    res_label_mat = res_dl_label_read(res_labels_list, fix_len)
    mask_vec = np.zeros([vec_mat.shape[0], fix_len])
    for i in range(len(label_len_list)):
        mask_vec[i, 0:label_len_list[i]] = 1
    return vec_mat, res_label_mat, mask_vec, label_len_list
    #特征矩阵为 vec_mat label 为res_label_mat

# 
def getFeatures16(fea_file, fix_len):
    vec_mat, fixed_seq_len_list = read_base_mat4res(fea_file, fix_len)
    return vec_mat
    #特征矩阵为 vec_mat label 为res_label_mat


# 生成测试特征

def read_base_mat4res_valid(in_file):
    # 从BLM/BioSeq-BLM/results/res_features.txt 中读取特征
    vectors_list = []
    seq_len_list = []
    f = open(in_file, 'r')
    lines = f.readlines()
    vectors = []
    flag = 0
    for line in lines:
        if len(line.strip()) != 0:
            if line[0] != '>':
                vector = line.strip().split('\t')
                vector = list(map(float, vector))
                vectors.append(vector)
                flag = 1
            else:
                if flag == 1:
                    seq_len_list.append(len(vectors))
                    vectors_list.append(torch.from_numpy(np.array(vectors)))
                    vectors = []
                    flag = 0
    f.close()
    vec_mat = vectors_list
    fixed_seq_len_list = seq_len_list
    return vec_mat, fixed_seq_len_list

#读取label
def res_dl_label_read_valid(res_label_list):
    res_label_mat = []
    for res_label in res_label_list:
        temp_arr = np.zeros(len(res_label))
        seq_len = len(res_label)
        temp_len = len(res_label)
        temp_arr[:temp_len] = res_label[:temp_len]
        res_label_mat.append(temp_arr)

    return np.array(res_label_mat)

#  生成label矩阵
def read_res_label_file_valid(label_file):
    
    res_labels_list = []
    label_len_list = []

    f = open(label_file, 'r')
    lines = f.readlines()
    for line in lines:
        if line[0] != '>':
            labels = line.strip().split()
            labels = list(map(int, labels))
            label_len_list.append(len(labels))
            res_labels_list.append(labels)
    f.close()

    return res_labels_list, label_len_list




def getFeaturesLabels_valid(fea_file, label_file):
    res_labels_list, label_len_list = read_res_label_file_valid(label_file)
    vec_mat, fixed_seq_len_list = read_base_mat4res_valid(fea_file)
    res_label_mat = res_dl_label_read_valid(res_labels_list)
    return vec_mat, res_label_mat
    #特征矩阵为 vec_mat label 为res_label_mat

def getFeatures16_valid(fea_file):
    vec_mat, fixed_seq_len_list = read_base_mat4res_valid(fea_file)
    return vec_mat