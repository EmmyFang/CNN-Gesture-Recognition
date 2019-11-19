'''
    The entry into your code. This file should include a training function and an evaluation function.

    # file name a_1 is the label
    # each file is 6 measurement
    # convert 42 x 130 files into two numpy arrays: feature and label
    # use conv1d
    # kernel: receptive field
    # size of kernel hyperparam
'''

import argparse
import string
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter

import torch
from torch.utils.data import DataLoader
from dataset import Gestures
from model import Net
import matplotlib.pyplot as plt

# instances_n = np.load('./data/normalized_data.npy')
instances_n = np.load('./data/instances.npy')
# instances_n = np.delete(instances_n,[2], axis = 2) # drop Az
# instances_n = np.delete(instances_n,[3], axis = 2) # drop Wx
# instances_n = np.delete(instances_n,[4], axis = 2) # drop Wy
labels = np.load('./data/labels.npy')
y = np.zeros(labels.shape, dtype = int)
list_alpha = list(string.ascii_lowercase)
for i in range(len(list_alpha)):
    y[labels == list_alpha[i]] = i

labels = y #convert labels from letters to integers


seed = 0 # random.randint(0, 1000)
test_size = 0.1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

X_train, X_val, y_train, y_val = train_test_split (instances_n, labels, test_size = test_size, random_state=seed)

def load_data(batch_size):
    train_dataset = Gestures(X_train, y_train)
    val_dataset = Gestures(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def load_model(lr, kernel_num, kernel_size, hidden_layer_size, op, hidden_layer_num, conv_layer_num, pooling_size, loss_func):
    if (loss_func == 'CE'):
        # "This criterion combines :func:`nn.LogSoftmax` and :func:`nn.NLLLoss` in one single class."
        loss_fnc = torch.nn.CrossEntropyLoss()

    else:
        # depricated
        loss_fnc = torch.nn.NLLLoss()
    model = Net(X_train.shape[2], kernel_num, kernel_size, hidden_layer_size, hidden_layer_num, conv_layer_num, pooling_size)
    if op == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else: # op == Adam
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, loss_fnc, optimizer


def evaluate(model, val_loader):
    total_corr = 0

    for i, vbatch in enumerate(val_loader):
        feats, label = vbatch
        prediction = model(feats)

        ind = torch.argmax(prediction, dim=1)
        corr = ind == label.long()
        total_corr += int(corr.sum())

    return float(total_corr)/len(val_loader.dataset)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=260)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default= 100)
    parser.add_argument('--eval_every', type=int, default=10)

    parser.add_argument('--activation_func', type=str, default='relu', help='activation_func, relu or tanh')
    parser.add_argument('--hidden_layer_num', type=int, default=1, help='select from 0 to 2')
    parser.add_argument('--hidden_layer_size', type=int, default=256)



    parser.add_argument('--conv_layer_num', type=int, default=2, help='select from 0 to 2')
    parser.add_argument('--kernel_num', type=int, default=64)
    parser.add_argument('--kernel_size', type=int, default=14)
    parser.add_argument('--pooling_size', type=int, default=4)

    parser.add_argument('--op', type=str, default='Adam', help = 'optimizer, SGD or Adam')
    parser.add_argument('--loss_func', type=str, default='CE', help='loss_func, CE or NLLLoss')


    args = parser.parse_args()


    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    eval_every = args.eval_every
    op = args.op
    kernel_size = args.kernel_size
    kernel_num = args.kernel_num
    hidden_layer_size = args.hidden_layer_size
    hidden_layer_num = args.hidden_layer_num
    conv_layer_num = args.conv_layer_num
    loss_func = args.loss_func
    activation_func = args.activation_func
    pooling_size = args.pooling_size


    val_acc_epoch = []
    train_loader, val_loader = load_data(batch_size)
    model, loss_fnc, optimizer = load_model(lr, kernel_num, kernel_size, hidden_layer_size, op, hidden_layer_num, conv_layer_num,
                                            pooling_size, loss_func)

    t = 0
    train_acc = []
    val_acc = []
    val_acc_step = []

    for epoch in range(epochs):
        accum_loss = 0
        tot_corr = 0
        train_corr = 0
        train_size_count = 0
        batch_num = 0

        for i, batch in enumerate(train_loader):
            feats, label = batch
            optimizer.zero_grad()
            predictions = model(feats)


            ind = torch.argmax(predictions, dim=1)

            a = predictions[0]
            b = predictions[1]
            if (loss_func == 'CE'):
                batch_loss = loss_fnc(input=predictions.squeeze(), target = label.long())

            else:
                batch_loss = loss_fnc(input=predictions.squeeze(), target=label.long())
            accum_loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()

            corr = ind.squeeze() == label.long()
            tot_corr += int(corr.sum())

            train_corr += int(corr.sum())
            train_size_count += len(label)
            batch_num += 1

            if (t + 1) % args.eval_every == 0:
                valid_acc = evaluate(model, val_loader)
                tr_acc = float(train_corr/train_size_count)
                print("Epoch: {}, Step {} | Loss: {}| val acc:{}| train acc: {}".format(epoch + 1, t + 1, accum_loss/batch_num, valid_acc,tr_acc ))
                val_acc.append(valid_acc)
                accum_loss = 0
                val_acc_step.append(t)
                train_acc.append(tr_acc)
                train_corr = 0
                train_size_count = 0
                batch_num = 0

            t = t + 1
        valid_acc = evaluate(model, val_loader)
        val_acc_epoch.append(valid_acc)
        print("Epoch: {}, Step {} | Loss: {}| val acc:{}".format(epoch + 1, t + 1, accum_loss/batch_num,valid_acc))
        torch.save(model, 'model_{}.pt'.format(epoch+1))
    model_path = "batch size = {}, lr = {}, epochs = {}, op= {}, hidden size= {}, hidden_layer_num={}, " \
                 "act_func= {}, kernel_size={}, kernel_num={}, conv_layer_num={}, loss_fnc={}, " \
                 "seed={}, pooling = {}".format(
        batch_size, lr, epochs, op, hidden_layer_size, hidden_layer_num, activation_func,kernel_size, kernel_num, conv_layer_num,
    loss_func, seed, pooling_size)
    print(model_path)
    print("Train acc:{}".format(float(tot_corr) / len(train_loader.dataset)))
    print("Max validation accuracy: ", max(val_acc))
    print("Max validation accuracy end of epoch: {}, at epoch {}".format(max(val_acc_epoch), val_acc_epoch.index(max(val_acc_epoch))+1))
    print("last validation accuracy: ", val_acc[-1])


    window = 11
    order = 5

    plt.figure()
    x = savgol_filter(val_acc, window, order)
    plt.plot(val_acc_step, x, label="validation accuracy")
    x = savgol_filter(train_acc, window, order)
    plt.plot(val_acc_step, x, label="train accuracy")
    plt.legend()
    plt.ylim([0, 1])
    plt.title("accuracy")
    plt.xlabel("steps")
    plt.ylabel("accuracy")
    plt.savefig("{}.png".format(model_path))
    plt.show()


    df_val = pd.DataFrame({"steps": val_acc_step, "val_acc": val_acc})
    df_val.to_csv("val_acc_{}.csv".format(model_path), index=False)
    df_train = pd.DataFrame({"steps": val_acc_step, "train_acc": train_acc})
    df_train.to_csv("train_acc_{}.csv".format(model_path), index=False)
if __name__ == "__main__":
    main()