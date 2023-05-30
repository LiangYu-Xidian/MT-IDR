import utils.utils_nogram
import torch
import warnings
import copy
warnings.filterwarnings("ignore")


# 给输入数据做normalization (做两种归一化的时候不确定分母是否会出现0 ，训练过程中数据出现NAN则需要进行处理)
def data_normal_2d(orign_data, dim):
    # orign_data shape = [seqence len, feature size]
    if(dim == "row"):  # 行归一化 在特征维度方向进行归一化
        min_row = orign_data.min(1, keepdim=True)[0]
        max_row = orign_data.max(1, keepdim=True)[0]
        norm_data = (orign_data - min_row) / (max_row - min_row)
    else:              # 列归一化
        min_col = orign_data.min(0, keepdim=True)[0]
        max_col = orign_data.max(0, keepdim=True)[0]
        norm_data = (orign_data - min_col) / (max_col - min_col)
    return norm_data

# 读取训练数据集数据
def get_train_data(fea_train_file, label_train_file, fix_len, dim):
    features_train ,label_train, mask, label_len_list = utils.utils_nogram.getFeaturesLabels(fea_train_file, label_train_file, fix_len)
    X_train =  torch.tensor(torch.from_numpy(features_train), dtype=torch.float32)
    Y_train = torch.tensor(torch.from_numpy(label_train), dtype=torch.float32)
    mask = torch.tensor(torch.from_numpy(mask), dtype=torch.float32)

    # 正则化数据 (注意训练集正则化需要注意对补0的部分的处理)
    for i in range(len(X_train)):
        length = label_len_list[i]         # 获取数据长度
        origin_data = X_train[i][0:length,:]  # 截断补零部分
        # 补零部分归一化
        norm_data = data_normal_2d(origin_data, dim)    
        X_train[i][0:length,:] = norm_data
    return X_train, Y_train, mask

# N 代表window_size 参考 morf
def get_train_data_window(fea_train_file, fea_train_file2, label_train_file, fix_len, dim):
    #获取PSSM特征并使用窗口捕获邻居信息
    N1 = 10
    N2 = 45
    N3 = 90
    features_train ,label_train, mask, label_len_list = utils.utils_nogram.getFeaturesLabels(fea_train_file, label_train_file, fix_len)
    X_train =  torch.tensor(torch.from_numpy(features_train), dtype=torch.float32)
    Y_train = torch.tensor(torch.from_numpy(label_train), dtype=torch.float32)
    mask = torch.tensor(torch.from_numpy(mask), dtype=torch.float32)
    X_train_multi = torch.zeros((X_train.shape[0], X_train.shape[1], X_train.shape[2] * 3))
    for i in range(len(X_train)):
        vec_mat1 = copy.deepcopy(X_train[i])
        vec_mat2 = copy.deepcopy(X_train[i])
        vec_mat3 = copy.deepcopy(X_train[i]) 
        temp1 = torch.zeros((X_train[i].shape[0]+ N1,X_train[i].shape[1]))
        temp2 = torch.zeros((X_train[i].shape[0]+ N2-1,X_train[i].shape[1]))
        temp3 = torch.zeros((X_train[i].shape[0]+ N3,X_train[i].shape[1]))
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
        X_train_multi[i,:,:] = v
    # 获取fea16 信息
    features_trian_fea16 = utils.utils_nogram.getFeatures16(fea_train_file2, fix_len)
    features_trian_fea16 = torch.tensor(features_trian_fea16)
    X_train = torch.cat([X_train_multi, features_trian_fea16], dim=2)
    # 正则化数据 (注意训练集正则化需要注意对补0的部分的处理)
    for i in range(len(X_train)):
        length = label_len_list[i]         # 获取数据长度
        origin_data = X_train[i][0:length,:]  # 截断补零部分
        # 补零部分归一化
        norm_data = data_normal_2d(origin_data, dim)    
        X_train[i][0:length,:] = norm_data

    return X_train, Y_train, mask

# 读取测试集数据
def get_test_data(fea_test_file, label_test_file, dim):
    X_test ,Y_test = utils.utils_nogram.getFeaturesLabels_valid(fea_test_file, label_test_file)
    
    # for i in range(len(X_test)):
    #     X_test[i] = data_normal_2d(X_test[i], dim)

    return X_test, Y_test

def get_test_data_window(fea_test_file, fea_test_file2, label_test_file, dim):
    # 获取fea16
    X_16 = utils.utils_nogram.getFeatures16_valid(fea_test_file2)
    # 获取PSSM
    X_test ,Y_test = utils.utils_nogram.getFeaturesLabels_valid(fea_test_file, label_test_file)
    N1 = 10
    N2 = 45
    N3 = 90
    temp = []
    for i in range(len(X_test)):
        vec_mat1 = copy.deepcopy(X_test[i])
        vec_mat2 = copy.deepcopy(X_test[i])
        vec_mat3 = copy.deepcopy(X_test[i]) 
        temp1 = torch.zeros((X_test[i].shape[0]+ N1,X_test[i].shape[1]))
        temp2 = torch.zeros((X_test[i].shape[0]+ N2-1,X_test[i].shape[1]))
        temp3 = torch.zeros((X_test[i].shape[0]+ N3,X_test[i].shape[1]))
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
        v = torch.cat([vec_mat1,vec_mat2,vec_mat3,X_16[i]], axis = 1)
        temp.append(v)

    X_test = temp
    for i in range(len(X_test)):
        X_test[i] = data_normal_2d(X_test[i], dim)
    

    return X_test, Y_test

# fea_train_morfs_file = "featuresAndLabels/res_features_train_morfs.txt"
# fea_train_morfs_file2 = "featuresAndLabels/features16/res_fea16_train_MoRFs.fasta"
# label_train_morfs_file = "featuresAndLabels/train_label_morfs.fasta"
# fix_len = 1500
# dim = "row"
# # idr数据
# fea_train_idr_file = "featuresAndLabels/res_features_train_idr.txt"
# fea_train_dir_file2 = "featuresAndLabels/features16/res_fea16_train_IDR.fasta"
# label_train_idr_file = "featuresAndLabels/train_label_idr.fasta"


# X_train_IDR, Y_train_IDR, mask_idr = get_train_data_window(fea_train_idr_file, fea_train_dir_file2, label_train_idr_file, fix_len, dim = dim)

# get_test_data_window(fea_train_morfs_file, fea_train_morfs_file2, label_train_morfs_file, dim)

# # 数据存放文件夹
# fea_train_morfs_file = "featuresAndLabels/res_features_train_morfs.txt"
# label_train_morfs_file = "featuresAndLabels/train_label_morfs.fasta"

# fea_file_morfs_test = "featuresAndLabels/res_features_test_morfs.txt"
# label_file_morfs_test = "featuresAndLabels/test_label_morfs.fasta"
# fea_file_morfs_test_test464 = "featuresAndLabels/Independent data/res_features_test464.txt"
# label_file_morfs_test_test464 = "featuresAndLabels/Independent data/test464_label.fasta"
# fea_file_morfs_test_EXP53 = "featuresAndLabels/Independent data/res_features_EXP53.txt"
# label_file_morfs_test_EXP53 = "featuresAndLabels/Independent data/EXP53_label.fasta"


# 获取数据
# X_train_MoRFs, Y_train_MoRFs, mask_MoRFs = get_train_data(fea_train_morfs_file, label_train_morfs_file, fix_len = 10, dim = "col")

# X_MoRFs_Test, Y_MoRFs_Test = get_test_data(fea_file_morfs_test, label_file_morfs_test, dim = "row")
# X_MoRFs_Test464, Y_MoRFs_Test464 = get_test_data(fea_file_morfs_test_test464, label_file_morfs_test_test464, dim = "col")
# X_MoRFs_EXP53, Y_MoRFs_EXP53 = get_test_data(fea_file_morfs_test_EXP53, label_file_morfs_test_EXP53, dim = "col")

# X_train_MoRFs, Y_train_MoRFs, mask_MoRFs = get_train_data_window(fea_train_morfs_file, label_train_morfs_file, fix_len = 10, dim = "col")
