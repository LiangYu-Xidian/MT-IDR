from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from model import Encoder
from MTL import MTLnet
import data_loader 
import torch
import numpy as np
import torch.nn as nn
import test
# 设置随机种子
def same_seeds(seed):
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    torch.backends.cudnn.benchmark = False  # GPU、网络结构固定，可设置为True
    torch.backends.cudnn.deterministic = True  # 固定网络结构

seed = 2
same_seeds(seed)

# 超参数
device = 'cpu'
learning_rate1 = 1e-5
learning_rate2 = 4 * 1e-5
n_epochs = 500
batch_size = 32
fix_len = 1500


# 模型参数
# Transformer Parameters
d_in_hidden = 64 # emebedding hidden size
d_model = 64  # Embedding Size
d_ff = 512 # FeedForward dimension
d_k = d_v = 32  # dimension of K(=Q), V
n_layers = 2  # number of Encoder of Decoder Layer
n_heads = 4  # number of heads in Multi-Head Attention
d_out_hidden = 32 # outputs hidden size
d_tower_hiden = 64
d_tower_hiden2 = 32
fea_dim = "row"  # data preprocessing - normalization
features_dim = 9 
# 加载模型
model = MTLnet(features_dim, d_in_hidden, d_model, d_ff, d_k, d_v, n_layers, n_heads, d_out_hidden, d_tower_hiden, d_tower_hiden2)
model = model.to(device)

#输出参数量
Trainable_params = 0
for param in model.parameters():
    mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
    if param.requires_grad:
        Trainable_params += mulValue  # 可训练参数量
print(f'Trainable params: {Trainable_params}')


# 选择 loss
loss_fn_1 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([20]))
loss_fn_2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([30]))
# 选择优化器
optimizer1 = optim.Adam(model.parameters(), lr=learning_rate1)
optimizer2 = optim.Adam(model.parameters(), lr=learning_rate2)

# 加载训练数据
# 数据存放文件夹
fea_train_morfs_file = "featuresAndLabels/res_features_train_morfs.txt"
fea_train_morfs_file2 = "featuresAndLabels/features16/res_fea16_train_MoRFs.fasta"
label_train_morfs_file = "featuresAndLabels/train_label_morfs.fasta"

fea_file_morfs_test = "featuresAndLabels/res_features_test_morfs.txt"
fea_file_morfs_test2 = "featuresAndLabels/features16/res_fea16_test_MoRFs.fasta"
label_file_morfs_test = "featuresAndLabels/test_label_morfs.fasta"
fea_file_morfs_test464 = "featuresAndLabels/Independent data/res_features_test464.txt"
fea_file_morfs_test464_2 = "featuresAndLabels/features16/res_fea16_test464_MoRFs.fasta"
label_file_morfs_test464 = "featuresAndLabels/Independent data/test464_label.fasta"
fea_file_morfs_EXP53 = "featuresAndLabels/Independent data/res_features_EXP53.txt"
fea_file_morfs_EXP53_2 = "featuresAndLabels/features16/res_fea16_exp53_MoRFs.fasta"
label_file_morfs_EXP53 = "featuresAndLabels/Independent data/EXP53_label.fasta"

# idr数据
fea_train_idr_file = "featuresAndLabels/res_features_train_idr.txt"
fea_train_dir_file2 = "featuresAndLabels/features16/res_fea16_train_IDR.fasta"
label_train_idr_file = "featuresAndLabels/train_label_idr.fasta"


X_train_IDR, Y_train_IDR, mask_idr = data_loader.get_train_data_window(fea_train_idr_file, fea_train_dir_file2, label_train_idr_file, fix_len, dim = fea_dim)
x_idr, y_idr = data_loader.get_test_data_window(fea_train_idr_file, fea_train_dir_file2, label_train_idr_file, dim = fea_dim)

X_train_MoRFs, Y_train_MoRFs, mask_morfs = data_loader.get_train_data_window(fea_train_morfs_file, fea_train_morfs_file2, label_train_morfs_file, fix_len, dim = fea_dim)
x_morfs,y_morfs = data_loader.get_test_data_window(fea_train_morfs_file, fea_train_morfs_file2, label_train_morfs_file, dim = fea_dim)

X_MoRFs_Test, Y_MoRFs_Test = data_loader.get_test_data_window(fea_file_morfs_test, fea_file_morfs_test2, label_file_morfs_test, dim = fea_dim)
X_MoRFs_Test464, Y_MoRFs_Test464 = data_loader.get_test_data_window(fea_file_morfs_test464,fea_file_morfs_test464_2, label_file_morfs_test464, dim = fea_dim)
X_MoRFs_EXP53, Y_MoRFs_EXP53 = data_loader.get_test_data_window(fea_file_morfs_EXP53,fea_file_morfs_EXP53_2, label_file_morfs_EXP53, dim = fea_dim)

X_train_IDR = X_train_IDR[:,:,99:]
X_train_MoRFs = X_train_MoRFs[:,:,99:]
for i in range(len(X_MoRFs_Test)):
    X_MoRFs_Test[i] = X_MoRFs_Test[i][:,99:]
for i in range(len(X_MoRFs_Test464)):
    X_MoRFs_Test464[i] = X_MoRFs_Test464[i][:,99:]
for i in range(len(X_MoRFs_EXP53)):
    X_MoRFs_EXP53[i] = X_MoRFs_EXP53[i][:,99:]
print(X_train_IDR.shape)
print(X_train_MoRFs.shape)
print(X_MoRFs_Test[0].shape)
print(X_MoRFs_Test464[0].shape)

def del_label2(y_true):                     # 获取数据中为2的元素
    index = [] 
    for i in range(y_true.shape[0]):
        index_1 = []
        for j in range(y_true[i].shape[0]):
            if(y_true[i][j] == 2):
                index_1.append(j)
        index.append(index_1)
    return index

index_TEST = del_label2(Y_MoRFs_Test)
index_TEST464 = del_label2(Y_MoRFs_Test464)
index_EXP53 = del_label2(Y_MoRFs_EXP53)

for i in range(Y_MoRFs_Test.shape[0]):
    Y_MoRFs_Test[i] = np.delete(Y_MoRFs_Test[i], index_TEST[i])
for i in range(Y_MoRFs_Test464.shape[0]):
    Y_MoRFs_Test464[i] = np.delete(Y_MoRFs_Test464[i], index_TEST464[i])
for i in range(Y_MoRFs_EXP53.shape[0]):
    Y_MoRFs_EXP53[i] = np.delete(Y_MoRFs_EXP53[i], index_EXP53[i])

# 将数据加载为DataLoader
def getTensorDataset(my_x, my_y, my_mask):
    tensor_x = torch.Tensor(my_x)
    tensor_y = torch.Tensor(my_y) 
    tensor_m = torch.Tensor(my_mask)
    return torch.utils.data.TensorDataset(tensor_x, tensor_y, tensor_m)
# 更改数据类型
X_train_IDR = torch.tensor(X_train_IDR, dtype=torch.float32)
train_loader_IDR = DataLoader(dataset=getTensorDataset(X_train_IDR, Y_train_IDR, mask_idr), batch_size=batch_size, shuffle=True)

X_train_MoRFs = torch.tensor(X_train_MoRFs, dtype=torch.float32)
train_loader_MoRFs = DataLoader(dataset=getTensorDataset(X_train_MoRFs, Y_train_MoRFs, mask_morfs), batch_size=batch_size, shuffle=True)


for epoch in range(n_epochs):
    model.train()
    loss_mean_idr = 0
    for x, y, mask in train_loader_IDR:
        XE, YE, MASKE  = x.to(device), y.to(device), mask.to(device)
        y1, y2 =  model(XE, MASKE, fix_len)
        y_hat = y1.squeeze(2)
        YE = YE*MASKE
        y_hat = MASKE*y_hat
        loss = loss_fn_1(y_hat, YE)
        loss_mean_idr += loss
        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()
    loss_mean_idr /=  int(X_train_MoRFs.shape[0] / batch_size + 0.5)
    
    loss_mean_morfs = 0
    for x, y, mask in train_loader_MoRFs:
        XE, YE, MASKE  = x.to(device), y.to(device), mask.to(device)
        y1, y2 =  model(XE, MASKE, fix_len)
        y_hat = y2.squeeze(2)
        YE = YE*MASKE
        y_hat = MASKE*y_hat
        loss = loss_fn_2(y_hat, YE)
        optimizer2.zero_grad()
        loss.backward()
        loss_mean_morfs += loss
        optimizer2.step()
    loss_mean_morfs /=  int(X_train_MoRFs.shape[0] / batch_size + 0.5)
    # 保存模型
    # mode_path = 'model/morf_12/' + str(epoch) + '.pth'
    # torch.save(model, mode_path) # 保存整个模型
    print(epoch)
    print("---------------train-----------------")
    print('IDR loss: {:.4}'.format(loss_mean_idr.item()))
    print('MoRFs loss: {:.4}'.format(loss_mean_morfs.item()))
    # test.test_train_data(model, loss_fn_2, x_morfs, y_morfs, 'MoRFs')
    print("---------------test------------------")
    test.test(model, loss_fn_2, X_MoRFs_Test, Y_MoRFs_Test, 'MoRFs', index_TEST)
    test.test(model, loss_fn_2, X_MoRFs_Test464, Y_MoRFs_Test464, "MoRFs", index_TEST464)
    test.test(model, loss_fn_2, X_MoRFs_EXP53, Y_MoRFs_EXP53, "EXP53", index_EXP53)


