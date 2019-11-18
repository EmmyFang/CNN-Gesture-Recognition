'''
    Save the data in the .csv file, save as a .npy file in ./data
'''

import numpy as np
import os


instance = []
label = []
for root, dirs, files in os.walk('./data/unnamed_train_data'):
    # print(root)
    for f in files:
        # print('files = ', f)
        filepath = root +'/'+ f
        # print(filepath)
        data = np.loadtxt(filepath,delimiter = ',')
        instance.append(np.delete(data,[0], axis = 1))
        label.append(f[0])
        # print(f[0])

instance = np.asarray(instance)
label = np.asarray(label)

print(instance.shape)
print(instance[0].shape)

print(label.shape)
print(label)

np.save('./data/instances',instance)
np.save('./data/labels', label)