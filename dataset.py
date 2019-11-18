'''
    Extend the torch.utils.data.Dataset class to build a GestureDataset class.
'''

import torch.utils.data as data


class Gestures(data.Dataset):

    def __init__(self, X, y):

        self.features = X
        self.label = y


    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        X = self.features[index]
        y = self.label[index]
        return X, y
