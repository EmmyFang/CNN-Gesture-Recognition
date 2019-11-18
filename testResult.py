import argparse
from time import time
import string
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import Gestures
from model import Net
import matplotlib.pyplot as plt
import random


# instances_n = np.delete(instances_n,[2], axis = 2) # drop Az
# instances_n = np.delete(instances_n,[3], axis = 2) # drop Wx
# instances_n = np.delete(instances_n,[4], axis = 2) # drop Wy

test_set = np.load('./assign3TestData/assign3part3/test_data.npy')
y = np.zeros(test_set.shape[0], dtype = int)
seed = 0 # random.randint(0, 1000)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

model = torch.load('model.pt')

def load_data(batch_size):
    test_dataset = Gestures(test_set,y)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader


def evaluate(model, val_loader):
    ind = 0

    for i, vbatch in enumerate(val_loader):
        feats, label = vbatch
        prediction = model(feats)

        ind = torch.argmax(prediction, dim=1)

    return ind

test_loader = load_data(test_set.shape[0])
result = evaluate(model,test_loader)
print(result)
np.savetxt("predictions.txt", result)


test_loader = load_data(260)
ind = []

for i, vbatch in enumerate(test_loader):
    feats, label = vbatch
    prediction = model(feats)

    ind.extend (np.asarray(torch.argmax(prediction, dim=1)))

y = np.asarray(result)
print(sum(y-ind))