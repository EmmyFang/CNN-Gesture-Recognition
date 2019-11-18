'''
    Visualize some samples.
'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_graph(path):
    plt.figure()
    data = np.loadtxt(path, delimiter=',')
    sample_number = np.arange(1,data.shape[0]+1,1)
    plt.title("sensor values vs sample number of {}_{}".format(path[-16:-8],path[-7:-4]))
    plt.plot(sample_number, data[:,1], label='Ax')
    plt.plot(sample_number, data[:,2], label='Ay')
    plt.plot(sample_number, data[:,3], label='Az')
    plt.plot(sample_number, data[:,4], label='Wx')
    plt.plot(sample_number, data[:,5], label='Wy')
    plt.plot(sample_number, data[:,6], label='Wz')
    plt.legend(loc='best')
    plt.xlabel("sensor value number")
    plt.ylabel('sensor value')
    plt.savefig("{}_{}.png".format(path[-16:-8], path[-7:-4]))

file = './data/unnamed_train_data/student0/a_1.csv'
plot_graph(file)
file = './data/unnamed_train_data/student1/a_1.csv'
plot_graph(file)
file = './data/unnamed_train_data/student2/a_1.csv'
plot_graph(file)

file = './data/unnamed_train_data/student1/o_1.csv'
plot_graph(file)
file = './data/unnamed_train_data/student2/o_1.csv'
plot_graph(file)
file = './data/unnamed_train_data/student4/o_1.csv'
plot_graph(file)