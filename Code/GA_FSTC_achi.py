import os
from torch.utils.data import Dataset
import torch
from torch import nn
from GA_FSTC import RFIDNetwork
from sklearn.metrics import confusion_matrix

device = torch.device('cuda')
alldata = []
labeldata = []
fenge = []
ztfenge = []
depth = 1
graph_args = {'strategy': 'distance'}
BATCH_SIZE = 64
label_tem = 0

def getdata(path):
    alldata = []
    maxdf, maxrssi, mindf = 0, 0, 0
    with open(path, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            x = line.split()
            a = float(x[3])
            b = float(x[4])
            if a > maxdf:
                maxdf = a
            if a < mindf:
                mindf = a
            if b < maxrssi:
                maxrssi = b
    with open(path, "r") as f:
        # print(path)
        for line in f.readlines():
            data = []
            line = line.strip('\n')
            x = line.split()
            data.append(int(x[1]))
            mmm = float(x[3])
            if mmm > 0:
                mmm = mmm / maxdf
            if mmm < 0:
                mmm = mmm / mindf
            xinh = float(x[4]) / maxrssi
            data.append(mmm)
            data.append(xinh)
            data.append(float(x[5]) / 6.280117345603815)
            alldata.append(data)
    newdata = []
    for k in range(16):
        newdata.append([])
    for j in alldata:
        a = j[0]
        newdata[a].append(j[1:4])
    tudata = []
    for i in newdata:
        danchongdao = []
        if len(i) <= 64:
            zerolist = [0] * 3
            for j in range(64 - len(i)):
                i.append(zerolist)
            tudata.append(i)
        if len(i) > 64:
            danchongdao = i[:64]
            tudata.append(danchongdao)
    reshapedata = []
    for i in range(64):
        a = []
        for j in range(16):
            a.append(tudata[j][i])
        reshapedata.append(a)
    return reshapedata

def data_lodar(path):
    datas = []
    labels = []
    path1 = os.listdir(path)
    for i in path1:
        path2 = os.path.join(path, str(i))
        path3 = os.listdir(path2)
        for j in path3:
            label = j.split('_')[1][:-4]
            labels.append(label)
            path4 = os.path.join(path2, j)
            data = getdata(path4)
            datas.append(data)
    print('label', len(labels), type(labels))
    print('data', len(datas), type(datas))
    return datas, labels

data, label = data_lodar('danrenshuju')

import random
def split_train_test(data, label, test_ratio):
    '''random.seed(38)
    random.shuffle(data)
    random.seed(38)
    random.shuffle(label)
    test_set_size = int(len(data) * test_ratio)
    print(test_set_size)
    test_data = torch.Tensor(data[:test_set_size])
    test_label = torch.Tensor(label[:test_set_size])
    train_data = torch.Tensor(data[test_set_size:])
    train_label = torch.Tensor(label[test_set_size:])'''
    random.seed(38)
    random.shuffle(data)
    random.seed(38)
    random.shuffle(label)
    test_set_size = int(len(data) * test_ratio)
    newdata, newlabel = [], []
    for k in range(21):
        newdata.append([]), newlabel.append([])
    for i in range(len(label)):
        newdata[int(label[i])].append(data[i])
    train_data, test_data, train_label, test_label = [], [], [], []
    for i in range(len(newdata)):
        for j in range(len(newdata[i])):
            if j < int(len(newdata[i]) * test_ratio):
                test_data.append(newdata[i][j])
                test_label.append(i)
            else:
                train_data.append(newdata[i][j])
                train_label.append(i)
    random.seed(38)
    random.shuffle(train_data)
    random.seed(38)
    random.shuffle(train_label)
    test_data = torch.Tensor(test_data)
    test_label = torch.Tensor(test_label)
    train_data = torch.Tensor(train_data)
    train_label = torch.Tensor(train_label)
    return train_data, train_label, test_data, test_label

traindata, trainlabel, testdata, testlabel = split_train_test(data, label, 0.2)
from torch.utils.data import Dataset, DataLoader, TensorDataset

print('4545', type(traindata))
traindata = TensorDataset(traindata, trainlabel)
testdata = TensorDataset(testdata, testlabel)
EPOCH = 1
TIME_STEP = 28
INPUT_SIZE = 28
LR = 0.001

def get_confusion_matrix(preds, labels, num_classes, normalize="true"):
    if isinstance(preds, list):
        preds = torch.cat(preds, dim=0)
    if isinstance(labels, list):
        labels = torch.cat(labels, dim=0)
    if labels.ndim == preds.ndim:
        labels = torch.argmax(labels, dim=-1)
    preds = torch.flatten(torch.argmax(preds, dim=-1))
    labels = torch.flatten(labels)
    cmtx = confusion_matrix(
        labels, preds, labels=list(range(num_classes)))
    return cmtx
rfid = RFIDNetwork(graph_args, 21).to(device)
optimizer = torch.optim.Adam(rfid.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()
train_loader = torch.utils.data.DataLoader(dataset=traindata, batch_size=BATCH_SIZE, shuffle=True)

def train(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        b_y = torch.tensor(b_y, dtype=torch.int64)
        b_x, b_y = b_x.to(device), b_y.to(device)
        output = rfid(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
best_acc = [0]
cmtxlist = []

def test(ep):
    test_loss = 0
    correct = 0
    preds = []
    labels = []
    test_data = DataLoader(testdata, batch_size=BATCH_SIZE, shuffle=False)
    if best_acc[-1] == max(best_acc):
        print('------------------------------------------更新参数----------------------------------------------')
        torch.save(rfid.state_dict(), 'GA_FSTC.mdl')
    with torch.no_grad():
        for data, target in test_data:
            target = torch.tensor(target, dtype=torch.int64)
            data, target = data.to(device), target.to(device)
            output = rfid(data).to(device)
            preds.append(output.cpu())
            labels.append(target.cpu())
            test_loss += loss_func(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        test_loss /= len(test_data.dataset)
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        cmtx = get_confusion_matrix(preds, labels, 21)
        print(cmtx)
        for i in range(len(cmtx)):
            print(cmtx[i][i]/sum(cmtx[i]))
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
            test_loss, correct, len(test_data.dataset),
            100. * correct / len(test_data.dataset)))
        acc = correct / len(test_data.dataset)
        best_acc.append(float(acc))
        cmtxlist.append(cmtx.tolist())
        if ep == 54:
            with open("GA_FSTC.txt", "a+") as f:
                f.write('\n')
                f.write(str(best_acc))
                f.write('\n')
                f.write(str(max(best_acc)))
                f.write(str(cmtxlist[best_acc.index(max(best_acc))]))
                f.write('\n')
        print('acc',best_acc)
        print(max(best_acc))
        return test_loss
import time
if __name__ == '__main__':
    for epoch in range(0, 55):
        print(epoch)
        train(epoch)
        Stime = time.time()
        test(epoch)
        Etime = time.time()
        print("rtime=", (Etime - Stime)/40)
        if epoch % 20 == 0:
            LR /= 10