import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn import metrics

def test_train_data(model, loss_fn, X_train_for_test, Y_train_for_test, s):
    model.eval()
    with torch.no_grad():
        Y_predict_train = np.zeros(1)
        Y_train = np.zeros(1)
        X_train_ = X_train_for_test
        Y_train_ = Y_train_for_test
        for k in range(len(X_train_)):
            temp = X_train_[k].float()
            temp = temp.unsqueeze(0)
            mask = torch.ones((1, temp.shape[1]))
            y1, y2 =  model(temp, mask, temp.shape[1])
            Yhat1M = y2.numpy()
            Yhat1M = Yhat1M.squeeze(axis = 0)
            Yhat1M = Yhat1M.squeeze(axis = 1)
            # 将预测结果进行 sigmod              （为了弥补BCELossWithLogitcs）
            Yhat1M = torch.from_numpy(Yhat1M)
            Yhat1M = torch.sigmoid(Yhat1M)
            Yhat1M = Yhat1M.numpy()
            Y_predict_train = np.concatenate([Y_predict_train, Yhat1M], axis=0)
            Y_train = np.concatenate([Y_train, Y_train_[k]], axis=0)
        Y_predict_train = Y_predict_train[1:]
        Y_train = Y_train[1:]
        AUC_T = roc_auc_score(Y_train, Y_predict_train)
        Y_predict = np.where(Y_predict_train>=0.5, 1, 0)
        BACC_T = metrics.balanced_accuracy_score(Y_train, Y_predict)
        MCC_T = metrics.matthews_corrcoef(Y_train, Y_predict)
        Pre_T = metrics.precision_score(Y_train, Y_predict)
        Recall_T = metrics.recall_score(Y_train, Y_predict)
        cm = metrics.confusion_matrix(Y_train, Y_predict)
        sn_T = cm[0][0]/(cm[0][0] + cm[0][1])
        sp_T = cm[1][1]/(cm[1][1] + cm[1][0])
        Y_predict_train = torch.tensor(Y_predict_train)
        Y_train = torch.tensor(Y_train)
        loss = loss_fn(Y_predict_train, Y_train)
        print( s +  ' loss: {:.4}'.format(loss.item()) + ' ' + s +  ' AUC: {:.4}'.format(AUC_T) + ' ' + s +  ' BACC: {:.4}'.format(BACC_T) +  ' ' + s +  ' MCC: {:.4}'.format(MCC_T) +  ' ' + s  +  ' Precision: {:.4}'.format(Pre_T) +  ' ' + s  +  ' Recall: {:.4}'.format(Recall_T) +  ' ' + s  +  ' sn: {:.4}'.format(sn_T) +  ' ' + s +  ' sp: {:.4}'.format(sp_T))


def test(model, loss_fn, X_train_for_test, Y_train_for_test, s, index):
    model.eval()
    with torch.no_grad():
        Y_predict_train = np.zeros(1)
        Y_train = np.zeros(1)
        X_train_ = X_train_for_test
        Y_train_ = Y_train_for_test
        for k in range(len(X_train_)):
            temp = X_train_[k].float()
            temp = temp.unsqueeze(0)
            mask = torch.ones((1, temp.shape[1]))
            y1, y2 =  model(temp, mask, temp.shape[1])
            Yhat1M = y2.numpy()
            Yhat1M = Yhat1M.squeeze(axis = 0)
            Yhat1M = Yhat1M.squeeze(axis = 1)
            # 将 预测结果 中包含2的mask掉
            Yhat1M = np.delete(Yhat1M, index[k])
            # 将预测结果进行 sigmod              （为了弥补BCELossWithLogitcs）
            Yhat1M = torch.from_numpy(Yhat1M)
            Yhat1M = torch.sigmoid(Yhat1M)
            Yhat1M = Yhat1M.numpy()
            Y_predict_train = np.concatenate([Y_predict_train, Yhat1M], axis=0)
            Y_train = np.concatenate([Y_train, Y_train_[k]], axis=0)
        Y_predict_train = Y_predict_train[1:]
        Y_train = Y_train[1:]
        AUC_T = roc_auc_score(Y_train, Y_predict_train)
        Y_predict = np.where(Y_predict_train>=0.5, 1, 0)
        BACC_T = metrics.balanced_accuracy_score(Y_train, Y_predict)
        MCC_T = metrics.matthews_corrcoef(Y_train, Y_predict)
        Pre_T = metrics.precision_score(Y_train, Y_predict)
        Recall_T = metrics.recall_score(Y_train, Y_predict)
        cm = metrics.confusion_matrix(Y_train, Y_predict)
        sn_T = cm[0][0]/(cm[0][0] + cm[0][1])
        sp_T = cm[1][1]/(cm[1][1] + cm[1][0])
        Y_predict_train = torch.tensor(Y_predict_train)
        Y_train = torch.tensor(Y_train)
        loss = loss_fn(Y_predict_train, Y_train)
        print( s +  ' loss: {:.4}'.format(loss.item()) + ' ' + s +  ' AUC: {:.4}'.format(AUC_T) + ' ' + s +  ' BACC: {:.4}'.format(BACC_T) +  ' ' + s +  ' MCC: {:.4}'.format(MCC_T) +  ' ' + s  +  ' Precision: {:.4}'.format(Pre_T) +  ' ' + s  +  ' Recall: {:.4}'.format(Recall_T) +  ' ' + s  +  ' sn: {:.4}'.format(sn_T) +  ' ' + s +  ' sp: {:.4}'.format(sp_T))

# 将预测结果保存下来
def test_save_res(model, loss_fn, X_train_for_test, Y_train_for_test, epoch, s, index):
    model.eval()
    with torch.no_grad():
        Y_predict_train = np.zeros(1)
        Y_train = np.zeros(1)
        X_train_ = X_train_for_test
        Y_train_ = Y_train_for_test
        for k in range(len(X_train_)):
            temp = X_train_[k].float()
            temp = temp.unsqueeze(0)
            mask = torch.ones((1, temp.shape[1]))
            y1, y2 =  model(temp, mask, temp.shape[1])
            Yhat1M = y2.numpy()
            Yhat1M = Yhat1M.squeeze(axis = 0)
            Yhat1M = Yhat1M.squeeze(axis = 1)
            # 将 label 中包含2的mask掉
            Yhat1M = np.delete(Yhat1M, index[k])
            # 将预测结果进行 sigmod              （为了弥补BCELossWithLogitcs）
            Yhat1M = torch.from_numpy(Yhat1M)
            Yhat1M = torch.sigmoid(Yhat1M)
            Yhat1M = Yhat1M.numpy()
            Y_predict_train = np.concatenate([Y_predict_train, Yhat1M], axis=0)
            Y_train = np.concatenate([Y_train, Y_train_[k]], axis=0)
           
        Y_predict_train = Y_predict_train[1:]
        Y_train = Y_train[1:]
        AUC_T = roc_auc_score(Y_train, Y_predict_train)
        Y_predict = np.where(Y_predict_train>=0.5, 1, 0)
        BACC_T = metrics.balanced_accuracy_score(Y_train, Y_predict)
        MCC_T = metrics.matthews_corrcoef(Y_train, Y_predict)
        Pre_T = metrics.precision_score(Y_train, Y_predict)
        Recall_T = metrics.recall_score(Y_train, Y_predict)
        cm = metrics.confusion_matrix(Y_train, Y_predict)
        sn_T = cm[0][0]/(cm[0][0] + cm[0][1])
        sp_T = cm[1][1]/(cm[1][1] + cm[1][0])
        Y_predict_train = torch.tensor(Y_predict_train)
        Y_train = torch.tensor(Y_train)
        loss = loss_fn(Y_predict_train, Y_train)
        file_dir = "res/" + s + str(epoch) + ".txt"
        np.savetxt(file_dir, Y_predict_train)
        print( s +  ' loss: {:.4}'.format(loss.item()) + ' ' + s +  ' AUC: {:.4}'.format(AUC_T) + ' ' + s +  ' BACC: {:.4}'.format(BACC_T) +  ' ' + s +  ' MCC: {:.4}'.format(MCC_T) +  ' ' + s  +  ' Precision: {:.4}'.format(Pre_T) +  ' ' + s  +  ' Recall: {:.4}'.format(Recall_T) +  ' ' + s  +  ' sn: {:.4}'.format(sn_T) +  ' ' + s +  ' sp: {:.4}'.format(sp_T))
