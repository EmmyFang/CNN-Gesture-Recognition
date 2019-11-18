'''
    Normalize the data, save as ./data/normalized_data.npy
'''

import numpy as np

instances = np.load('./data/instances.npy')
labels = np.load('./data/labels.npy')

for sample in range(instances.shape[0]): # 5590 samples

    for measure in range (instances.shape[2]): # 6 measures each sample

        mean = instances[sample,:,measure].mean()
        std = instances[sample,:,measure].std()
        instances[sample, :, measure] = (instances[sample, :, measure] - mean)/std


np.save('./data/normalized_data',instances)