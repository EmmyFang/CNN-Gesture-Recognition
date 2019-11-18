'''
    Visualize some basic statistics of our dataset.
'''

import matplotlib.pyplot as plt
import numpy as np
import string

def autolabel(ax, rects, fontsize=14):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1*height,'%f' % height,
                ha='center', va='bottom',fontsize=fontsize)

def binary_bar_chart(mean, std, i):

    ind = np.arange(len(mean))  # the x locations for the groups

    width = 0.40
    fig, ax = plt.subplots(figsize=(12, 7))
    above_bars = ax.bar(ind, mean, width, color='#41f474', yerr = std)

    ax.set_xlabel("Types", fontsize=20)
    ax.set_ylabel('Mean measurement ', fontsize=20)
    ax.set_title('Measurement of {}'.format(i), fontsize=22)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('Ax', 'Ay', 'Az', 'Wx', 'Wy', 'Wz'),
                       fontsize=7)

    autolabel(ax, above_bars)
    plt.savefig('bar_{}'.format(i))
    plt.show()


instances = np.load('./data/instances.npy')
labels = np.load('./data/labels.npy')
list_alpha = list(string.ascii_lowercase)
all_mean = []
all_sd = []
for letter in list_alpha:
    index = labels == letter
    gesture = instances[index]
    letter_mean = []
    letter_sd = []
    for measure in range (gesture.shape[2]): # traverse through ax to wz
        gest_collapse = gesture[:,:,measure].flatten()
        letter_mean.append(gest_collapse.mean())
        letter_sd.append(gest_collapse.std())


    all_mean.append(letter_mean)
    all_sd.append(letter_sd)

all_mean = np.asarray(all_mean).reshape(26,1,6)
all_sd = np.asarray(all_sd).reshape(26,1,6)

for i in range (2):
    binary_bar_chart(all_mean[i,0,:], all_sd[i,0,:],list_alpha[i])




