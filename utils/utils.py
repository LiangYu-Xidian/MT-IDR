
import random
import numpy as np
import torch
import copy
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

def getFeaturesLabels(fea_file, label_file, fix_len, type_):
    res_labels_list, label_len_list = read_res_label_file(label_file)
    vec_mat, fixed_seq_len_list = read_base_mat4res(fea_file, fix_len)
    res_label_mat = res_dl_label_read(res_labels_list, fix_len)
    mask_vec = np.zeros([vec_mat.shape[0], fix_len])
    for i in range(len(label_len_list)):
        mask_vec[i, 0:label_len_list[i]] = 1
    vec_mat1 = copy.deepcopy(vec_mat)
    vec_mat2 = copy.deepcopy(vec_mat)
    vec_mat3 = copy.deepcopy(vec_mat)    
    N1 = 10
    N2 = 45
    N3 = 90
    temp1 = np.zeros((vec_mat.shape[0],vec_mat.shape[1] + N1,vec_mat.shape[2]))
    temp2 = np.zeros((vec_mat.shape[0],vec_mat.shape[1] + N2-1,vec_mat.shape[2]))
    temp3 = np.zeros((vec_mat.shape[0],vec_mat.shape[1] + N3,vec_mat.shape[2]))
    temp1[:,int(N1/2):temp1.shape[1]-int(N1/2),:] = vec_mat1
    temp2[:,int(N2/2):temp2.shape[1]-int(N2/2),:] = vec_mat2
    temp3[:,int(N3/2):temp3.shape[1]-int(N3/2),:] = vec_mat3
    for i in range(temp1.shape[1] - N1):
        f = temp1[:,i:i+N1,:]
        vec_mat1[:,i,:] = f.mean(axis = 1)
    for i in range(temp2.shape[1] - N2):
        f = temp2[:,i:i+N2,:]
        vec_mat2[:,i,:] = f.mean(axis = 1)
    for i in range(temp3.shape[1] - N3):
        f = temp3[:,i:i+N3,:]
        vec_mat3[:,i,:] = f.mean(axis = 1)
    vec_mat = np.concatenate([vec_mat1,vec_mat2,vec_mat3],axis=2)
    max_morfs = np.array([ 8.88000011e-01,  2.45770001e+00,  1.28760014e+01,  1.66999984e+00,1.46960011e+01,  1.26400006e+00,  9.64000106e-01, -2.50999965e-02,1.50999999e+00,  1.27400017e+00,  1.35299993e+00,  2.65899992e+00,2.45770001e+00,  7.20000000e-01,  4.91000000e+00,  1.50000000e+01,1.67000000e+00,  1.46300000e+01,  1.58000000e+00,  8.93777788e-01,1.00000000e-03,  1.51000000e+00,  1.35000000e+00,  1.23000000e+00, 2.64000000e+00,  4.91000000e+00,  4.21666712e-01,  1.22286677e+00, 1.23156652e+01,  8.71222258e-01,  1.38530006e+01,  1.10622215e+00, 8.79555583e-01, -5.22000007e-02,  1.26655567e+00,  1.09866667e+00,1.13466656e+00,  1.98155546e+00 , 1.22286677e+00])
    min_morfs = np.array([-2.28999996, -2.24000001,  5.26599979, -0.71200001,  5.73199987,  0.35600001,0.30699998, -0.41100001,  0.31300002,  0.28000003,  0.41100001,  0.,-2.24000001, -2.29 ,      -2.24,        5.49977779, -0.77,        6.09622192, 0.37,        0.2,        -0.411,       0.36399999,  0.33288887,  0.40044445, 0.,         -2.24 ,      -1.32788897, -0.98731107,  4.46922207, -0.15599999, 5.06344414,  0.36699998,  0.32822219, -0.37208891,  0.40466663,  0.36211106, 0.37055555 , 0.23199999, -0.98731107])
    max_idr = np.array([ 0.824,       2.40209985, 13.41600037  ,1.41000009 ,14.94999981 , 1.54999995,0.95299995,  0.06340001 , 1.523  ,     1.45300007 , 1.44200003 , 2.67199993, 2.40209985,  1.01   ,     4.91  ,     15.   ,       1.67     ,  15.36, 1.58 ,       0.97   ,     0.275 ,      1.53  ,      1.58    ,    1.54,2.86 ,       4.91   ,     0.70666665,  1.2680223,  12.06344414,  0.81699997,13.41522217,  1.16911113,  0.88300002, -0.04876667,  1.28277779,  1.29600012, 1.32277775,  2.50855541,  1.2680223 ])
    min_idr = np.array([-1.84200001, -1.33999991 , 5.1500001 , -0.65300006,  5.63299942,  0.34600002,0.24100001, -0.38240001 , 0.296  ,     0.27000001 , 0.36000001,  0.,-1.33999991, -2.29 ,      -2.24 ,       5.41222191, -0.77    ,    5.93644476,0.37,        0.2 ,       -0.411  ,     0.3764444 ,  0.36222219,  0.42622223,0. ,        -2.24   ,    -0.9423334 , -0.88684446,  5.44922209, -0.10377776,5.87711096 , 0.42133331,  0.373  ,    -0.3127889,   0.40755555,  0.37911111,0.41333336 , 0.34977776, -0.88684446])
    if(type_ == "idr"):
        vec_mat = (vec_mat - max_idr) / (max_idr - min_idr)
    if(type_ == "morfs"):
        vec_mat = (vec_mat - min_morfs) / (max_morfs - min_morfs)
    # 对idr数据和morf数据直接做归一化
    return vec_mat, res_label_mat, mask_vec, fixed_seq_len_list
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

def getFeaturesLabels_valid(fea_file, label_file, type_):
    res_labels_list, label_len_list = read_res_label_file_valid(label_file)
    vec_mat, fixed_seq_len_list = read_base_mat4res_valid(fea_file)
    res_label_mat = res_dl_label_read_valid(res_labels_list)
    vec_mat_temp = []
    N1 = 10
    N2 = 45
    N3 = 90
    max_morfs = np.array([ 8.88000011e-01,  2.45770001e+00,  1.28760014e+01,  1.66999984e+00,1.46960011e+01,  1.26400006e+00,  9.64000106e-01, -2.50999965e-02,1.50999999e+00,  1.27400017e+00,  1.35299993e+00,  2.65899992e+00,2.45770001e+00,  7.20000000e-01,  4.91000000e+00,  1.50000000e+01,1.67000000e+00,  1.46300000e+01,  1.58000000e+00,  8.93777788e-01,1.00000000e-03,  1.51000000e+00,  1.35000000e+00,  1.23000000e+00, 2.64000000e+00,  4.91000000e+00,  4.21666712e-01,  1.22286677e+00, 1.23156652e+01,  8.71222258e-01,  1.38530006e+01,  1.10622215e+00, 8.79555583e-01, -5.22000007e-02,  1.26655567e+00,  1.09866667e+00,1.13466656e+00,  1.98155546e+00 , 1.22286677e+00])
    min_morfs = np.array([-2.28999996, -2.24000001,  5.26599979, -0.71200001,  5.73199987,  0.35600001,0.30699998, -0.41100001,  0.31300002,  0.28000003,  0.41100001,  0.,-2.24000001, -2.29 ,      -2.24,        5.49977779, -0.77,        6.09622192, 0.37,        0.2,        -0.411,       0.36399999,  0.33288887,  0.40044445, 0.,         -2.24 ,      -1.32788897, -0.98731107,  4.46922207, -0.15599999, 5.06344414,  0.36699998,  0.32822219, -0.37208891,  0.40466663,  0.36211106, 0.37055555 , 0.23199999, -0.98731107])
    max_idr = np.array([ 0.824,       2.40209985, 13.41600037  ,1.41000009 ,14.94999981 , 1.54999995,0.95299995,  0.06340001 , 1.523  ,     1.45300007 , 1.44200003 , 2.67199993, 2.40209985,  1.01   ,     4.91  ,     15.   ,       1.67     ,  15.36, 1.58 ,       0.97   ,     0.275 ,      1.53  ,      1.58    ,    1.54,2.86 ,       4.91   ,     0.70666665,  1.2680223,  12.06344414,  0.81699997,13.41522217,  1.16911113,  0.88300002, -0.04876667,  1.28277779,  1.29600012, 1.32277775,  2.50855541,  1.2680223 ])
    min_idr = np.array([-1.84200001, -1.33999991 , 5.1500001 , -0.65300006,  5.63299942,  0.34600002,0.24100001, -0.38240001 , 0.296  ,     0.27000001 , 0.36000001,  0.,-1.33999991, -2.29 ,      -2.24 ,       5.41222191, -0.77    ,    5.93644476,0.37,        0.2 ,       -0.411  ,     0.3764444 ,  0.36222219,  0.42622223,0. ,        -2.24   ,    -0.9423334 , -0.88684446,  5.44922209, -0.10377776,5.87711096 , 0.42133331,  0.373  ,    -0.3127889,   0.40755555,  0.37911111,0.41333336 , 0.34977776, -0.88684446])
    for i in range(len(vec_mat)):
        vec_mat1 = copy.deepcopy(vec_mat[i])
        vec_mat2 = copy.deepcopy(vec_mat[i])
        vec_mat3 = copy.deepcopy(vec_mat[i]) 
        temp1 = torch.zeros((vec_mat[i].shape[0]+ N1,vec_mat[i].shape[1]))
        temp2 = torch.zeros((vec_mat[i].shape[0]+ N2-1,vec_mat[i].shape[1]))
        temp3 = torch.zeros((vec_mat[i].shape[0]+ N3,vec_mat[i].shape[1]))
        temp1[int(N1/2):temp1.shape[0]-int(N1/2),:] = vec_mat1
        temp2[int(N2/2):temp2.shape[0]-int(N2/2),:] = vec_mat2
        temp3[int(N3/2):temp3.shape[0]-int(N3/2),:] = vec_mat3
        for k in range(temp1.shape[0] - N1):
            f = temp1[k:k+N1,:]
            vec_mat1[k,:] = f.mean(axis = 0)
        for k in range(temp2.shape[0] - N2):
            f = temp2[k:k+N2,:]
            vec_mat2[k,:] = f.mean(axis = 0)
        for k in range(temp3.shape[0] - N3):
            f = temp3[k:k+N3,:]
            vec_mat3[k,:] = f.mean(axis = 0)
        v = torch.cat([vec_mat1,vec_mat2,vec_mat3], axis = 1)
        if(type_ == "idr"):
            v = (v - max_idr) / (max_idr - min_idr)
        if(type_ == "morfs"):
            v = (v - min_morfs) / (max_morfs - min_morfs)
        vec_mat_temp.append(v)
    return vec_mat_temp, res_label_mat
    #特征矩阵为 vec_mat label 为res_label_mat