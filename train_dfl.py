from syslog import LOG_EMERG
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
loss_fn_2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4.5]))
# 选择优化器
optimizer1 = optim.Adam(model.parameters(), lr=learning_rate1)
optimizer2 = optim.Adam(model.parameters(), lr=learning_rate2)

# 加载训练数据
# 数据存放文件夹
fea_train_dfl_file = "featuresAndLabels/res_features_train_dfl.txt"
fea_train_dfl_file2 = "featuresAndLabels/features16/res_fea16_train_dfl.fasta"
label_train_dfl_file = "featuresAndLabels/train_label_dfl.fasta"


fea_file_test_dfl82 = "featuresAndLabels/res_features_test_dfl82.txt"
fea_file_test2_dfl82 = "featuresAndLabels/features16/res_fea16_test_dfl82.fasta"
label_file_test_dfl82 = "featuresAndLabels/test_label_dfl82.fasta"

fea_file_test_dfl64 = "featuresAndLabels/res_features_test_dfl64.txt"
fea_file_test2_dfl64 = "featuresAndLabels/features16/res_fea16_test_dfl64.fasta"
label_file_test_dfl64 = "featuresAndLabels/test_label_dfl64.fasta"


# idr数据
fea_train_idr_file = "featuresAndLabels/res_features_train_idr.txt"
fea_train_idr_file2 = "featuresAndLabels/features16/res_fea16_train_IDR.fasta"
label_train_idr_file = "featuresAndLabels/train_label_idr.fasta"



X_train_IDR, Y_train_IDR, mask_idr = data_loader.get_train_data_window(fea_train_idr_file, fea_train_idr_file2, label_train_idr_file, fix_len, dim = fea_dim)
x_idr, y_idr = data_loader.get_test_data_window(fea_train_idr_file, fea_train_idr_file2, label_train_idr_file, dim = fea_dim)
X_train_DFL, Y_train_DFL, mask_dfl = data_loader.get_train_data_window(fea_train_dfl_file, fea_train_dfl_file2, label_train_dfl_file, fix_len, dim = fea_dim)
x_dfl,y_dfl = data_loader.get_test_data_window(fea_train_dfl_file, fea_train_dfl_file2, label_train_dfl_file, dim = fea_dim)
X_DFL_Test82, Y_DFL_Test82 = data_loader.get_test_data_window(fea_file_test_dfl82, fea_file_test2_dfl82, label_file_test_dfl82, dim = fea_dim)
X_DFL_Test64, Y_DFL_Test64 = data_loader.get_test_data_window(fea_file_test_dfl64, fea_file_test2_dfl64, label_file_test_dfl64, dim = fea_dim)

###########
X_train_IDR = X_train_IDR[:,:,99:]
X_train_DFL = X_train_DFL[:,:,99:]
for i in range(len(X_DFL_Test82)):
    X_DFL_Test82[i] = X_DFL_Test82[i][:,99:]
for i in range(len(X_DFL_Test64)):
    X_DFL_Test64[i] = X_DFL_Test64[i][:,99:]
print(X_train_IDR.shape)
print(X_train_DFL.shape)
print(X_DFL_Test82[0].shape)
print(X_DFL_Test64[0].shape)


def del_label2(y_true):                     # 获取数据中为2的元素
    index = [] 
    for i in range(y_true.shape[0]):
        index_1 = []
        for j in range(y_true[i].shape[0]):
            if(y_true[i][j] == 2):
                index_1.append(j)
        index.append(index_1)
    return index

index_82 = del_label2(Y_DFL_Test82)
index_64 = del_label2(Y_DFL_Test64)

for i in range(Y_DFL_Test82.shape[0]):
    Y_DFL_Test82[i] = np.delete(Y_DFL_Test82[i], index_82[i])

for i in range(Y_DFL_Test64.shape[0]):
    Y_DFL_Test64[i] = np.delete(Y_DFL_Test64[i], index_64[i])

# 将数据加载为DataLoader
def getTensorDataset(my_x, my_y, my_mask):
    tensor_x = torch.Tensor(my_x)
    tensor_y = torch.Tensor(my_y) 
    tensor_m = torch.Tensor(my_mask)
    return torch.utils.data.TensorDataset(tensor_x, tensor_y, tensor_m)
# 更改数据类型
X_train_IDR = torch.tensor(X_train_IDR, dtype=torch.float32)
train_loader_IDR = DataLoader(dataset=getTensorDataset(X_train_IDR, Y_train_IDR, mask_idr), batch_size=batch_size, shuffle=True)

########################################
X_train_DFL = torch.tensor(X_train_DFL, dtype=torch.float32)
train_loader_DFL = DataLoader(dataset=getTensorDataset(X_train_DFL, Y_train_DFL, mask_dfl), batch_size=batch_size, shuffle=True)

n_epochs = 300
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
        optimizer1.zero_grad()
        loss.backward()
        loss_mean_idr += loss
        optimizer1.step()
    loss_mean_idr /=  int(X_train_DFL.shape[0] / batch_size + 0.5)
    loss_mean_dfl = 0
    for x, y, mask in train_loader_DFL:
        XE, YE, MASKE  = x.to(device), y.to(device), mask.to(device)
        y1, y2 =  model(XE, MASKE, fix_len)
        y_hat = y2.squeeze(2)
        YE = YE*MASKE
        y_hat = MASKE*y_hat
        loss = loss_fn_2(y_hat, YE)
        optimizer2.zero_grad()
        loss.backward()
        loss_mean_dfl += loss
        optimizer2.step()
    loss_mean_dfl /=  int(X_train_DFL.shape[0] / batch_size + 0.5)
    print(epoch)
    # mode_path = 'model/DFL_5/' + str(epoch) + '.pth'
    # torch.save(model, mode_path) # 保存整个模型
    print("---------------train-----------------")
    print('IDR loss: {:.4}'.format(loss_mean_idr.item()))
    print('DFL loss: {:.4}'.format(loss_mean_dfl.item()))
    # test.test_train_data(model, loss_fn_2, x_dfl, y_dfl, "DFL")
    print("---------------test------------------")
    test.test(model, loss_fn_2,  X_DFL_Test82, Y_DFL_Test82, "DFL82 ", index_82)
    test.test(model, loss_fn_2,  X_DFL_Test64, Y_DFL_Test64, "DFL64 ", index_64)

