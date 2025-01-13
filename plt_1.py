import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
import time

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'arial'

def smooth(input):

    x1 = (input[0] + input[1] + input[2]) / 3
    x2 = (input[0] + input[1] + input[3]) / 3
    x3 = (input[0] + input[2] + input[3]) / 3
    x4 = (input[1] + input[2] + input[3]) / 3
    output = np.array([x1, x3, x2, x4])

    return output

def moving_average(input_data, window_size):
    moving_average = [[] for i in range(len(input_data))]
    for i in range(len(input_data)):
        for j in range(len(input_data[i])):
            if j < window_size - 1:
                if type(input_data[i][j + 1]) == str:
                    input_data[i][j + 1] = float(input_data[i][j + 1])
                # print(i, j, input_data[i][:j + 1])
                moving_average[i].append(sum(input_data[i][:j + 1]) / len(input_data[i][:j + 1]))
            else:
                # print(input_data[i][j - window_size + 1:j + 1])
                moving_average[i].append(sum(input_data[i][j - window_size + 1:j + 1]) / len(input_data[i][j - window_size + 1:j + 1]))
    moving_average_means = []
    for i in range(len(moving_average[0])):
        sum_data = []
        for j in range(len(moving_average)):
            sum_data.append(moving_average[j][i])
        moving_average_means.append(sum(sum_data) / len(sum_data))
    # print(len(moving_average_means))
    return np.array(moving_average), moving_average_means


# plot_list = ['EFD|0.0|4|False|.txt', 'CHOCO|0.0|4|False|.txt', 'DCD|0.0|4|False|.txt', 'ECD|0.0|4|False|.txt']
# plot_list = ['EFD|0.1|4|False|.txt', 'CHOCO|0.1|4|False|.txt', 'DCD|0.1|4|False|.txt', 'ECD|0.1|4|False|.txt']
# plot_list = ['EFD|0.0|6|False|.txt', 'CHOCO|0.0|6|False|.txt', 'DCD|0.0|6|False|.txt', 'ECD|0.0|6|False|.txt']
# plot_list = ['EFD|0.1|6|False|.txt', 'CHOCO|0.1|6|False|.txt', 'DCD|0.1|6|False|.txt', 'ECD|0.1|6|False|.txt']
# plot_list = ['EFD|0.0|8|False|.txt', 'CHOCO|0.0|8|False|.txt', 'DCD|0.0|8|False|.txt', 'ECD|0.0|8|False|.txt']
# plot_list = ['EFD|0.1|8|False|.txt', 'CHOCO|0.1|8|False|.txt', 'DCD|0.1|8|False|.txt', 'ECD|0.1|8|False|.txt']
# plot_list = ['EFD|0.0|10|False|.txt', 'CHOCO|0.0|10|False|.txt', 'DCD|0.0|10|False|.txt', 'ECD|0.0|10|False|.txt']
# plot_list = ['EFD|0.1|10|False|.txt', 'CHOCO|0.1|10|False|.txt', 'DCD|0.1|10|False|.txt', 'ECD|0.1|10|False|.txt']

# plot_list = ['EFD|0.0|0.4|False|1.0|1.txt', 'CHOCO|0.0|0.4|False|1.0|.txt', 'DCD|0.0|0.4|False|1.0|.txt', 'ECD|0.0|0.4|False|1.0|.txt']
# plot_list = ['EFD|0.1|0.4|False|.txt', 'CHOCO|0.1|0.4|False|.txt', 'DCD|0.1|0.4|False|.txt', 'ECD|0.1|0.4|False|.txt']

# plot_list = ['EFD|8|0|.txt', 'CHOCO|8|0|.txt', 'DCD|8|0|.txt', 'ECD|8|0|.txt']
# plot_list = ['EFD|8|0.1|.txt', 'CHOCO|8|0.1|.txt', 'DCD|8|0.1|.txt', 'ECD|8|0.1|.txt']
# plot_list = ['EFD|0.1|0|.txt', 'CHOCO|0.1|0|.txt', 'DCD|0.1|0|.txt', 'ECD|0.1|0|.txt']
# plot_list = ['EFD|0.1|0.1|.txt', 'CHOCO|0.1|0.1|.txt', 'DCD|0.1|0.1|.txt', 'ECD|0.1|0.1|.txt']

# plot_list = ['EFDwd|0.1|0|1|.txt', 'CHOCO|0.1|0|1|.txt', 'DCD|0.1|0|1|.txt', 'ECD|0.1|0|1|.txt']
# plot_list = ['EFDwd|0.1|0.1|1|.txt', 'CHOCO|0.1|0.1|1|.txt', 'DCD|0.1|0.1|1|.txt', 'ECD|0.1|0.1|1|.txt']
# plot_list = ['EFDwd|0.2|0|.txt', 'CHOCO|0.2|0|.txt', 'DCD|0.2|0|.txt', 'ECD|0.2|0|.txt']
# plot_list = ['EFDwd|0.2|0.1|.txt', 'CHOCO|0.2|0.1|.txt', 'DCD|0.2|0.1|.txt', 'ECD|0.2|0.1|.txt']

# plot_list = ['EFDwd|4|0|.txt', 'CHOCO|4|0|.txt', 'DCD|4|0|.txt', 'ECD|4|0|.txt']
# plot_list = ['EFDwd|4|0.1|.txt', 'CHOCO|4|0.1|.txt', 'DCD|4|0.1|.txt', 'ECD|4|0.1|.txt']
# plot_list = ['EFDwd|6|0|.txt', 'CHOCO|6|0|.txt', 'DCD|6|0|.txt', 'ECD|6|0|.txt']
# plot_list = ['EFDwd|6|0.1|.txt', 'CHOCO|6|0.1|.txt', 'DCD|6|0.1|.txt', 'ECD|6|0.1|.txt']
# plot_list = ['EFDwd|8|0|1|.txt', 'CHOCO|8|0|1|.txt', 'DCD|8|0|1|.txt', 'ECD|8|0|1|.txt']
# plot_list = ['EFDwd|8|0.1|1|.txt', 'CHOCO|8|0.1|1|.txt', 'DCD|8|0.1|1|.txt', 'ECD|8|0.1|1|.txt']

# plot_list = ['EFDwd|8|0|c|.txt', 'CHOCO|8|0|c|.txt', 'DCD|8|0|c|.txt', 'ECD|8|0|c|.txt']
# plot_list = ['EFDwd|8|0.1|c|.txt', 'CHOCO|8|0.1|c|.txt', 'DCD|8|0.1|c|.txt', 'ECD|8|0.1|c|.txt']
# plot_list = ['EFDwd|0.1|0|c|.txt', 'CHOCO|0.1|0|c|.txt', 'DCD|0.1|0|c|.txt', 'ECD|0.1|0|c|.txt']
# plot_list = ['EFDwd|0.1|0.1|c|.txt', 'CHOCO|0.1|0.1|c|.txt', 'DCD|0.1|0.1|c|.txt', 'ECD|0.1|0.1|c|.txt']

# plot_list = ['EFD|0|0.05|0.0|.txt', 'EFD|0|0.05|0.2|.txt', 'EFD|0|0.05|0.4|.txt', 'EFD|0|0.05|0.6|.txt', 'EFD|0|0.05|0.8|.txt', 'EFD|0|0.05|1.0|.txt']
# plot_list = ['EFD|0|0.3|0.0|.txt', 'EFD|0|0.3|0.3|.txt', 'EFD|0|0.3|0.6|.txt', 'EFD|0|0.3|0.8|.txt', 'EFD|0|0.3|0.9|.txt', 'EFD|0|0.3|1.0|.txt']

# plot_list = ['EFD|0|0.1|0.0|.txt', 'EFD|0|0.1|0.3|.txt', 'EFD|0|0.1|0.6|.txt', 'EFD|0|0.1|0.8|.txt', 'EFD|0|0.1|0.9|.txt', 'EFD|0|0.1|1.0|.txt']
# plot_list = ['EFD|0|0.2|0.0|1.txt', 'EFD|0|0.2|0.3|1.txt', 'EFD|0|0.2|0.6|1.txt', 'EFD|0|0.2|0.8|1.txt', 'EFD|0|0.2|0.9|1.txt', 'EFD|0|0.2|1.0|1.txt']

# plot_list = ['EFD|0|4|0.0|.txt', 'EFD|0|4|0.3|.txt', 'EFD|0|4|0.6|.txt', 'EFD|0|4|0.8|.txt', 'EFD|0|4|0.9|.txt', 'EFD|0|4|1.0|.txt']
# plot_list = ['EFD|0|8|0.0|.txt', 'EFD|0|8|0.3|.txt', 'EFD|0|8|0.6|.txt', 'EFD|0|8|0.8|.txt', 'EFD|0|8|0.9|.txt', 'EFD|0|8|1.0|.txt']

# plot_list = ['EFD|0.0|0.2|0.0|.txt', 'EFD|0.0|0.2|0.2|.txt', 'EFD|0.0|0.2|0.4|.txt', 'EFD|0.0|0.2|0.6|.txt', 'EFD|0.0|0.2|0.8|.txt', 'EFD|0.0|0.2|0.9|.txt']
# plot_list = ['EFD|0.0|8|0.0|.txt', 'EFD|0.0|8|0.2|.txt', 'EFD|0.0|8|0.4|.txt', 'EFD|0.0|8|0.6|.txt', 'EFD|0.0|8|0.8|.txt', 'EFD|0.0|8|1.0|.txt']
# plot_list = ['EFD|0.0|0.1|0.0|.txt', 'EFD|0.0|0.1|0.2|.txt', 'EFD|0.0|0.1|0.4|.txt', 'EFD|0.0|0.1|0.6|.txt', 'EFD|0.0|0.1|0.8|.txt']
# plot_list = ['EFD|0.0|10|0.0|.txt', 'EFD|0.0|10|0.2|.txt', 'EFD|0.0|10|0.4|.txt', 'EFD|0.0|10|0.6|.txt', 'EFD|0.0|10|0.8|.txt', 'EFD|0.0|10|1.0|.txt']
# plot_list = ['EFD|0.0|0.05|0.0|CIFAR.txt', 'EFD|0.0|0.05|0.2|CIFAR.txt', 'EFD|0.0|0.05|0.4|CIFAR.txt', 'EFD|0.0|0.05|0.6|CIFAR.txt', 'EFD|0.0|0.05|0.8|CIFAR.txt']

# plot_list = ['EFD|0.05|0.1|0.0|4.txt', 'EFD|0.05|0.1|0.2|4.txt', 'EFD|0.05|0.1|0.4|4.txt', 'EFD|0.05|0.1|0.6|4.txt', 'EFD|0.05|0.1|0.8|4.txt', 'EFD|0.05|0.1|1.0|4.txt']
# plot_list = ['EFD|0.05|0.2|0.0|4.txt', 'EFD|0.05|0.2|0.2|4.txt', 'EFD|0.05|0.2|0.4|4.txt', 'EFD|0.05|0.2|0.6|4.txt', 'EFD|0.05|0.2|0.8|4.txt', 'EFD|0.05|0.2|1.0|4.txt']
# plot_list = ['EFD|0.05|4|0.0|4.txt', 'EFD|0.05|4|0.2|4.txt', 'EFD|0.05|4|0.4|4.txt', 'EFD|0.05|4|0.6|4.txt', 'EFD|0.05|4|0.8|4.txt', 'EFD|0.05|4|1.0|4.txt']
# plot_list = ['EFD|0.05|6|0.0|4.txt', 'EFD|0.05|6|0.2|4.txt', 'EFD|0.05|6|0.4|4.txt', 'EFD|0.05|6|0.6|4.txt', 'EFD|0.05|6|0.8|4.txt', 'EFD|0.05|6|1.0|4.txt']
# plot_list = ['EFD|0.05|8|0.0|4.txt', 'EFD|0.05|8|0.2|4.txt', 'EFD|0.05|8|0.4|4.txt', 'EFD|0.05|8|0.6|4.txt', 'EFD|0.05|8|0.8|4.txt', 'EFD|0.05|8|1.0|4.txt']

# plot_list = ['EFD|0.05|0.1|0.0|lr.txt', 'EFD|0.05|0.1|0.1|lr.txt', 'EFD|0.05|0.1|0.2|lr.txt', 'EFD|0.05|0.1|0.3|lr.txt',
#              'EFD|0.05|0.1|0.4|lr.txt', 'EFD|0.05|0.1|0.5|lr.txt', 'EFD|0.05|0.1|0.6|lr.txt', 'EFD|0.05|0.1|0.7|lr.txt',
#              'EFD|0.05|0.1|0.8|lr.txt', 'EFD|0.05|0.1|0.9|lr.txt', 'EFD|0.05|0.1|1.0|lr.txt']
#
# plot_list = ['EFD|0.05|0.2|0.0|lr.txt', 'EFD|0.05|0.2|0.1|lr.txt', 'EFD|0.05|0.2|0.2|lr.txt', 'EFD|0.05|0.2|0.3|lr.txt',
#              'EFD|0.05|0.2|0.4|lr.txt', 'EFD|0.05|0.2|0.5|lr.txt', 'EFD|0.05|0.2|0.6|lr.txt', 'EFD|0.05|0.2|0.7|lr.txt',
#              'EFD|0.05|0.2|0.8|lr.txt', 'EFD|0.05|0.2|0.9|lr.txt', 'EFD|0.05|0.2|1.0|lr.txt']
#
# plot_list = ['EFD|0.05|4|0.0|lr.txt', 'EFD|0.05|4|0.1|lr.txt', 'EFD|0.05|4|0.2|lr.txt', 'EFD|0.05|4|0.3|lr.txt',
#              'EFD|0.05|4|0.4|lr.txt', 'EFD|0.05|4|0.5|lr.txt', 'EFD|0.05|4|0.6|lr.txt', 'EFD|0.05|4|0.7|lr.txt',
#              'EFD|0.05|4|0.8|lr.txt', 'EFD|0.05|4|0.9|lr.txt', 'EFD|0.05|4|1.0|lr.txt']
#
# plot_list = ['EFD|0.05|8|0.0|lr.txt', 'EFD|0.05|8|0.1|lr.txt', 'EFD|0.05|8|0.2|lr.txt', 'EFD|0.05|8|0.3|lr.txt',
#              'EFD|0.05|8|0.4|lr.txt', 'EFD|0.05|8|0.5|lr.txt', 'EFD|0.05|8|0.6|lr.txt', 'EFD|0.05|8|0.7|lr.txt',
#              'EFD|0.05|8|0.8|lr.txt', 'EFD|0.05|8|0.9|lr.txt', 'EFD|0.05|8|1.0|lr.txt']

# plot_list = ['EFD|0.05|8|0.0|lr.txt', 'EFD|0.05|8|0.1|lr.txt', 'EFD|0.05|8|0.2|lr.txt', 'EFD|0.05|8|0.3|lr.txt',
#              'EFD|0.05|8|0.4|lr.txt', 'EFD|0.05|8|0.5|lr.txt', 'EFD|0.05|8|0.6|lr.txt', 'EFD|0.05|8|0.7|lr.txt',
#              'EFD|0.05|8|0.8|lr.txt', 'EFD|0.05|8|0.9|lr.txt', 'EFD|0.05|8|1.0|lr.txt', 'EFD|0.05|8|0.01|lr.txt']

# plot_list = ['CHOCO|0.05|0.1|.txt', 'EFD|0.05|0.1|0.0|lr.txt', 'EFD|0.05|0.1|0.8|lr.txt']
# plot_list = ['CHOCO|0.05|0.2|.txt', 'EFD|0.05|0.2|0.0|lr.txt', 'EFD|0.05|0.2|0.9|lr.txt']
# plot_list = ['CHOCO|0.05|4|.txt', 'EFD|0.05|4|0.0|lr.txt', 'EFD|0.05|4|1.0|lr.txt']
# plot_list = ['CHOCO|0.05|8|.txt', 'EFD|0.05|8|0.0|lr.txt', 'EFD|0.05|8|0.3|lr.txt']

# plot_list = ['CHOCO|0|0.1|.txt', 'EFD|0|0.1|0.0|lr.txt', 'EFD|0|0.1|0.8|lr.txt']
# plot_list = ['CHOCO|0|0.2|.txt', 'EFD|0|0.2|0.0|lr.txt', 'EFD|0|0.2|0.8|lr.txt']
# plot_list = ['CHOCO|0|4|.txt', 'EFD|0|4|0.0|lr.txt', 'EFD|0|4|1.0|lr.txt']
# plot_list = ['CHOCO|0|8|.txt', 'EFD|0|8|0.0|lr.txt', 'EFD|0|8|1.0|lr.txt']

# plot_list = ['EFD|0.05|0.1|0.0|CIFARdc.txt', 'EFD|0.05|0.1|0.2|CIFARdc.txt', 'EFD|0.05|0.1|0.4|CIFARdc.txt',
#              'EFD|0.05|0.1|0.6|CIFARdc.txt', 'EFD|0.05|0.1|0.8|CIFARdc.txt', 'CHOCO|0.05|0.1|CIFARdc.txt']

# plot_list = ['EFD|0.05|0.2|0.0|CIFARdc.txt', 'EFD|0.05|0.2|0.2|CIFARdc.txt', 'EFD|0.05|0.2|0.4|CIFARdc.txt',
#              'EFD|0.05|0.2|0.6|CIFARdc.txt', 'EFD|0.05|0.2|0.8|CIFARdc.txt', 'CHOCO|0.05|0.2|CIFARdc.txt']

# plot_list = ['EFD|0.05|0.1|0.0|CIFARdc.txt', 'EFD|0.05|0.1|0.2|CIFARdc.txt', 'EFD|0.05|0.1|0.4|CIFARdc.txt',
#              'EFD|0.05|0.1|0.6|CIFARdc.txt', 'EFD|0.05|0.1|0.8|CIFARdc.txt']

# plot_list = ['EFD|0.05|0.2|0.0|CIFARdc.txt', 'EFD|0.05|0.2|0.2|CIFARdc.txt', 'EFD|0.05|0.2|0.4|CIFARdc.txt',
#              'EFD|0.05|0.2|0.6|CIFARdc.txt', 'EFD|0.05|0.2|0.8|CIFARdc.txt']
#
# plot_list = ['EFD|0.05|8|0.0|CIFARdc.txt', 'EFD|0.05|8|0.95|CIFARdc.txt',
#              'EFD|0.05|8|0.96|CIFARdc.txt', 'EFD|0.05|8|0.97|CIFARdc.txt', 'EFD|0.05|8|0.98|CIFARdc.txt',
#              'EFD|0.05|8|0.99|CIFARdc.txt', 'EFD|0.05|8|1.0|CIFARdc.txt', 'CHOCO|0.05|8|CIFARdc.txt']
#
# plot_list = ['CHOCO|0.0|0.1|1000.txt', 'EFD|0.0|0.1|0.0|1000.txt', 'EFD|0.0|0.1|0.8|1000.txt']
# plot_list = ['CHOCO|0.0|0.2|1000.txt', 'EFD|0.0|0.2|0.0|1000.txt', 'EFD|0.0|0.2|0.9|1000.txt']

# plot_list = ['EFD|0.0|4|1.0|1000.txt', 'EFD|0.1|4|1.0|1000.txt']
# plot_list = ['EFD|0.0|8|1.0|1000.txt', 'EFD|0.1|8|1.0|1000.txt']

# plot_list = ['EFD|0.05|8|0.0|lr.txt', 'EFD|0.05|8|0.01|lr.txt']

# plot_list = ['EFD|0.01|0.05|0.0|.txt', 'EFD|0.01|0.05|0.1|.txt', 'EFD|0.01|0.05|0.2|.txt', 'EFD|0.01|0.05|0.3|.txt',
#              'EFD|0.01|0.05|0.4|.txt', 'EFD|0.01|0.05|0.5|.txt', 'EFD|0.01|0.05|0.6|.txt', 'EFD|0.01|0.05|0.7|.txt',
#              'EFD|0.01|0.05|0.8|.txt', 'EFD|0.01|0.05|0.9|.txt', 'EFD|0.01|0.05|1.0|.txt']
#
# plot_list = ['EFD|0.01|0.1|0.0|.txt', 'EFD|0.01|0.1|0.1|.txt', 'EFD|0.01|0.1|0.2|.txt', 'EFD|0.01|0.1|0.3|.txt',
#              'EFD|0.01|0.1|0.4|.txt', 'EFD|0.01|0.1|0.5|.txt', 'EFD|0.01|0.1|0.6|.txt', 'EFD|0.01|0.1|0.7|.txt',
#              'EFD|0.01|0.1|0.8|.txt', 'EFD|0.01|0.1|0.9|.txt', 'EFD|0.01|0.1|1.0|.txt']
#
# plot_list = ['EFD|0.01|4|0.0|.txt', 'EFD|0.01|4|0.1|.txt', 'EFD|0.01|4|0.2|.txt', 'EFD|0.01|4|0.3|.txt',
#              'EFD|0.01|4|0.4|.txt', 'EFD|0.01|4|0.5|.txt', 'EFD|0.01|4|0.6|.txt', 'EFD|0.01|4|0.7|.txt',
#              'EFD|0.01|4|0.8|.txt', 'EFD|0.01|4|0.9|.txt', 'EFD|0.01|4|1.0|.txt']
#
# plot_list = ['EFD|0.01|8|0.0|.txt', 'EFD|0.01|8|0.1|.txt', 'EFD|0.01|8|0.2|.txt', 'EFD|0.01|8|0.3|.txt',
#              'EFD|0.01|8|0.4|.txt', 'EFD|0.01|8|0.5|.txt', 'EFD|0.01|8|0.6|.txt', 'EFD|0.01|8|0.7|.txt',
#              'EFD|0.01|8|0.8|.txt', 'EFD|0.01|8|0.9|.txt', 'EFD|0.01|8|1.0|.txt']
#
# plot_list = ['EFD|0.05|8|0.0|lr.txt', 'EFD|0.05|8|0.001|1.txt', 'EFD|0.05|8|0.0001|1.txt', 'EFD|0.05|8|0.00001|1.txt',
#              'EFD|0.05|8|0.1|lr.txt', 'EFD|0.05|8|0.2|lr.txt', 'EFD|0.05|8|0.3|lr.txt', 'EFD|0.05|8|0.4|lr.txt',
#              'EFD|0.05|8|0.9|lr.txt', 'EFD|0.05|8|1.0|lr.txt']
#
# plot_list = ['EFD|0.05|8|0.0|comp.txt', 'EFD|0.05|8|0.1|comp.txt', 'EFD|0.05|8|0.01|comp.txt', 'EFD|0.05|8|0.001|comp.txt',
#              'EFD|0.05|8|0.0001|comp.txt']
#
# plot_list = ['EFD|0.05|8|0.0|comp.txt', 'EFD|0.05|8|0.1|comp.txt', 'EFD|0.05|8|0.2|comp.txt', 'EFD|0.05|8|0.3|comp.txt', 'EFD|0.05|8|0.4|comp.txt',
#              'EFD|0.05|8|0.5|comp.txt', 'EFD|0.05|8|0.6|comp.txt', 'EFD|0.05|8|0.7|comp.txt', 'EFD|0.05|8|0.8|comp.txt',
#              'EFD|0.05|8|0.9|comp.txt', 'EFD|0.05|8|1.0|comp.txt']

# plot_list = ['EFD|0.05|8|0.8|CIFARdc.txt', 'EFD|0.05|8|0.9|CIFARdc.txt', 'EFD|0.05|8|0.95|CIFARdc.txt',
#              'EFD|0.05|8|0.96|CIFARdc.txt', 'EFD|0.05|8|0.97|CIFARdc.txt', 'EFD|0.05|8|0.98|CIFARdc.txt',
#              'EFD|0.05|8|0.99|CIFARdc.txt', 'EFD|0.05|8|1.0|CIFARdc.txt']

# plot_list = ['EFD|0.0|10|0.0|CIFAR.txt', 'EFD|0.0|10|0.2|CIFAR.txt', 'EFD|0.0|10|0.4|CIFAR.txt', 'EFD|0.0|10|0.6|CIFAR.txt',
#              'EFD|0.0|10|0.8|CIFAR.txt', 'EFD|0.0|10|1.0|CIFAR.txt']

# plot_list = ['EFD|0.05|0.1|0.0|lr.txt', 'EFD|0.05|0.1|0.8|lr.txt', 'CHOCO|0.05|0.1|.txt']
# plot_list = ['EFD|0.05|0.2|0.0|lr.txt', 'EFD|0.05|0.2|0.9|lr.txt', 'CHOCO|0.05|0.2|.txt']
# plot_list = ['EFD|0.05|4|0.0|lr.txt', 'EFD|0.05|4|1.0|lr.txt', 'CHOCO|0.05|4|.txt']
# plot_list = ['EFD|0.05|8|0.0|lr.txt', 'EFD|0.05|8|1.0|lr.txt', 'CHOCO|0.05|8|.txt']

# plot_list = ['CHOCO|0.05|0.1|0.0|CIFARdc4.txt', 'EFD|0.05|0.1|0.0|CIFARdc4.txt', 'EFD|0.05|0.1|0.6|CIFARdc4.txt']

"08-16-2024 compare with different baseline algorithms"
# plot_list = ['EFD|0.05|4|1.0|.txt', 'EFD|0.05|4|1.01|.txt', 'EFD|0.05|4|1.02|.txt', 'EFD|0.05|4|1.03|.txt', 'EFD|0.05|4|1.04|.txt',
#              'EFD|0.05|4|1.1|.txt', 'EFD|0.05|4|1.2|.txt', 'EFD|0.05|4|1.3|.txt', 'EFD|0.05|4|1.4|.txt']
#
# plot_list = ['EFD|0.05|4|1.0|.txt', 'EFD|0.05|4|1.001|.txt', 'EFD|0.05|4|1.002|.txt', 'EFD|0.05|4|1.003|.txt', 'EFD|0.05|4|1.004|.txt',
#              'EFD|0.05|4|1.005|.txt', 'EFD|0.05|4|1.006|.txt', 'EFD|0.05|4|1.007|.txt', 'EFD|0.05|4|1.008|.txt', 'EFD|0.05|4|1.009|.txt']

# plot_list = ['EFD|0.05|8|1.0|.txt', 'EFD|0.05|8|1.1|.txt', 'EFD|0.05|8|1.2|.txt', 'EFD|0.05|8|1.3|.txt', 'EFD|0.05|8|1.4|.txt',
#              'EFD|0.05|8|1.5|.txt', 'EFD|0.05|8|1.6|.txt', 'EFD|0.05|8|1.7|.txt']

# plot_list = ['CHOCO|0.05|0.1|0.0|CIFAR4.txt', 'EFD|0.05|0.1|0.0|CIFAR4.txt', 'EFD|0.05|0.1|0.6|CIFAR4.txt']
# plot_list = ['CHOCO|0.05|0.2|0.0|CIFAR4.txt', 'EFD|0.05|0.2|0.0|CIFAR4.txt', 'EFD|0.05|0.2|0.6|CIFAR4.txt']
# plot_list = ['CHOCO|0.05|8|0.0|CIFAR4.txt', 'EFD|0.05|8|0.0|CIFAR4.txt', 'EFD|0.05|8|1.0|CIFAR4.txt']
# plot_list = ['CHOCO|0.05|10|0.0|CIFAR4.txt', 'EFD|0.05|10|0.0|CIFAR4.txt', 'EFD|0.05|10|1.0|CIFAR4.txt']

# plot_list = ['CHOCO|0.05|0.1|.txt', 'EFD|0.05|0.1|0.0|lr.txt', 'EFD|0.05|0.1|0.8|lr.txt']
# plot_list = ['CHOCO|0.05|0.2|.txt', 'EFD|0.05|0.2|0.0|lr.txt', 'EFD|0.05|0.2|0.9|lr.txt']
# plot_list = ['CHOCO|0.05|4|.txt', 'EFD|0.05|4|0.0|lr.txt', 'EFD|0.05|4|1.0|lr.txt']
# plot_list = ['CHOCO|0.05|8|.txt', 'EFD|0.05|8|0.0|lr.txt', 'EFD|0.05|8|0.3|lr.txt']

"08/25/2024 Unbiased Quantization"

# plot_list = ['CHOCO|0.05|4|0.0|.txt', 'AdaG|0.05|4|0.0|lr.txt', 'QSADDLe|0.05|4|1.0|1.txt', 'DCD|0.05|4|0.0|.txt', 'EFD|0.05|4|1.0|.txt']
# plot_list = ['CHOCO|0.05|6|1.0|.txt', 'AdaG|0.05|6|0.0|lr.txt', 'QSADDLe|0.05|6|1.0|1.txt', 'DCD|0.05|6|0.0|.txt', 'EFD|0.05|6|1.0|.txt']

"CIFAR10"
# plot_list = ['CHOCO|0.05|6|0.0|CIFAR10.txt', 'AdaG|0.05|6|0.0|False|CIFAR10|0.01|0.004|.txt', 'DCD|0.05|6|0.0|CIFAR10.txt', 'EFD|0.05|6|1.0|CIFAR10.txt']
# plot_list = ['CHOCO|0.05|6|0.0|CIFAR10.txt', 'DCD|0.05|8|0.0|CIFAR10.txt', 'EFD|0.05|8|1.0|CIFAR10.txt']
# plot_list = ['CHOCO|0.05|0.1|0.0|CIFAR10.txt', 'AdaG|0.05|0.1|0.0|False|CIFAR10|0.056|0.002|.txt', 'DCD|0.05|0.1|0.0|CIFAR4.txt', 'EFD|0.05|0.1|0.6|CIFAR4.txt']
# plot_list = ['CHOCO|0.05|0.2|0.0|CIFAR4.txt', 'AdaG|0.05|0.2|0.0|False|CIFAR10|0.056|0.003|.txt', 'DCD|0.05|0.2|0.0|CIFAR4.txt', 'EFD|0.05|0.2|0.6|CIFAR4.txt']

"FashionMNIST topk"  # AdaG has the problem with consensus for top-k compression
# plot_list = ['CHOCO|0.05|0.1|0.0|.txt', 'AdaG|0.05|0.1|0.0|0.056.txt', 'QSADDLe|0.05|0.1|0.4|.txt', 'DCD|0.05|0.1|0.0|.txt', 'EFD|0.05|0.1|0.6|1.txt']
# plot_list = ['CHOCO|0.05|0.2|0.0|.txt', 'AdaG|0.05|0.2|0.0|0.056.txt', 'QSADDLe|0.05|0.2|0.5|.txt', 'DCD|0.05|0.2|0.0|.txt', 'EFD|0.05|0.2|0.9|1.txt']

"Compare the influence from gamma"
# plot_list = ['EFD|0.05|0.1|0.1|0.056.txt', 'EFD|0.05|0.1|0.2|0.056.txt', 'EFD|0.05|0.1|0.3|0.056.txt', 'EFD|0.05|0.1|0.4|0.056.txt',
#              'EFD|0.05|0.1|0.5|0.056.txt', 'EFD|0.05|0.1|0.6|0.056.txt', 'EFD|0.05|0.1|0.7|0.056.txt', 'EFD|0.05|0.1|0.8|0.056.txt',
#              'EFD|0.05|0.1|0.9|0.056.txt', 'EFD|0.05|0.1|1.0|0.056.txt']
#
# plot_list = ['CHOCO|0.05|0.1|0.1|.txt', 'CHOCO|0.05|0.1|0.2|.txt', 'CHOCO|0.05|0.1|0.3|.txt',
#              'CHOCO|0.05|0.1|0.4|.txt', 'CHOCO|0.05|0.1|0.5|.txt', 'CHOCO|0.05|0.1|0.6|.txt', 'CHOCO|0.05|0.1|0.7|.txt']

"Different Dirichlet parameter alpha = 0.5"
# plot_list = ['CHOCO|0.5|0.1|0.0|0.056|0.1|.txt', 'EFDwd|0.5|0.1|0.0|c.txt', 'EFDwd|0.5|0.1|0.8|0.1.txt']
# plot_list = ['CHOCO|0.5|0.2|0.0|0.056|0.2|.txt', 'EFDwd|0.5|0.2|0.0|0.056.txt', 'EFDwd|0.5|0.2|0.9|0.056.txt']
# plot_list = ['CHOCO|0.5|4|0.0|0.056|0.2|.txt', 'EFD|0.5|4|0.0|0.056.txt', 'EFD|0.5|4|0.9|0.1.txt']
# plot_list = ['CHOCO|0.5|6|0.0|0.056|0.2|.txt', 'EFD|0.5|6|0.0|0.056.txt', 'EFDwd|0.5|6|0.9|0.056.txt']

"Dirichlet parameter alpha = 0.01"
# plot_list = ['CHOCO|0.01|0.1|0.0|0.056|0.2|.txt', 'EFD|0.01|0.1|0.0|0.0316.txt', 'EFD|0.01|0.1|0.7|0.056.txt']
# plot_list = ['CHOCO|0.01|0.2|0.0|0.056|0.2|.txt', 'EFD|0.01|0.2|0.0|0.0316.txt', 'EFD|0.01|0.2|0.5|0.056.txt']
# plot_list = ['CHOCO|0.01|4|0.0|0.056|0.2|.txt', 'EFD|0.01|4|0.0|0.0316.txt', 'EFD|0.01|4|1.0|0.056.txt']
# plot_list = ['CHOCO|0.01|6|0.0|0.056|0.2|.txt', 'EFD|0.01|6|0.0|0.0316.txt', 'EFD|0.01|6|1.0|0.056.txt']

"Example to show normal error feedback not working"
# plot_list = ['CHOCO|0.05|0.1|0.0|0.5|0.056.txt', 'EFD|0.05|0.1|0.0|0.056.txt', 'EFD|0.05|0.1|1.0|0.0561.txt', 'EFD|0.05|0.1|0.6|0.0561.txt']

"Ring Network topology"
# plot_list = ['CHOCO|0.05|4|0.0|0.056|Ring.txt', 'DCD|0.05|4|0.0|0.056|Ring.txt', 'EFD|0.05|4|1.0|0.056|Ring.txt']
# plot_list = ['CHOCO|0.05|6|0.0|0.056|Ring.txt', 'DCD|0.05|6|0.0|0.056|Ring.txt', 'EFD|0.05|6|1.0|0.056|Ring.txt']
# plot_list = ['CHOCO|0.05|0.1|0.0|0.056|Ring.txt', 'DCD|0.05|0.1|0.0|0.056|Ring.txt', 'EFD|0.05|0.1|0.8|0.056|Ring.txt', 'AdaG|0.05|0.1|1.0|0.056|0.01|Ring.txt']
# plot_list = ['CHOCO|0.05|0.2|0.0|0.056|Ring.txt', 'DCD|0.05|0.2|0.0|0.056|Ring.txt', 'EFD|0.05|0.2|0.9|0.056|Ring.txt', 'AdaG|0.05|0.2|1.0|0.056|0.02|Ring.txt']

# plot_list = ['CHOCO|0.05|0.1|0.0|CIFAR10|0.056|0.2|Ring|.txt', 'DCD|0.05|0.1|0.0|CIFAR10|0.056|Ring|.txt', 'EFD|0.05|0.1|0.6|CIFAR10|0.056|Ring|.txt']
# plot_list = ['CHOCO|0.05|6|0.0|CIFAR10|0.056|0.2|Ring|.txt', 'DCD|0.05|6|0.0|CIFAR10|0.056|Ring|.txt', 'EFD|0.05|6|1.0|CIFAR10|0.056|Ring|.txt']

"Number of neighbors equals to 3"
# plot_list = ['CHOCO|0.05|4|0.0|0.056|0.2|.txt', 'DCD|0.05|4|0.0|0.056.txt', 'EFD|0.05|4|0.9|0.056.txt']
# plot_list = ['CHOCO|0.05|6|0.0|0.056|0.2|.txt', 'DCD|0.05|6|0.0|0.056.txt', 'EFD|0.05|6|0.9|0.056.txt']
# plot_list = ['CHOCO|0.05|0.1|0.0|0.056|0.2|.txt', 'DCD|0.05|0.1|0.0|0.056.txt', 'EFD|0.05|0.1|0.8|0.056.txt', 'EFD|0.05|0.1|1.0|0.5|3.txt']
# plot_list = ['CHOCO|0.05|0.2|0.0|0.056|0.2|.txt', 'DCD|0.05|0.2|0.0|0.056.txt', 'EFD|0.05|0.2|0.9|0.056.txt']

"Number of neighbors equals to 5"
# plot_list = ['CHOCO|0.05|0.1|0.0|0.056|0.2|5.txt', 'DCD|0.05|0.1|0.0|0.056|5.txt', 'EFD|0.05|0.1|0.7|0.056|5.txt', 'EFD|0.05|0.1|1.0|0.5|5.txt']
# plot_list = ['CHOCO|0.05|0.2|0.0|0.056|0.2|5.txt', 'DCD|0.05|0.2|0.0|0.056|5.txt', 'EFD|0.05|0.2|0.8|0.056|5.txt']
# plot_list = ['CHOCO|0.05|4|0.0|0.056|0.2|5.txt', 'DCD|0.05|4|0.0|0.056|5.txt', 'EFD|0.05|4|1.0|0.056|5.txt']
# plot_list = ['CHOCO|0.05|6|0.0|0.056|0.2|5.txt', 'DCD|0.05|6|0.0|0.056|5.txt', 'EFD|0.05|6|0.9|0.056|5.txt']

"Test"
# plot_list = ['CHOCO|0.05|0.1|0.0|.txt', 'DCD|0.05|0.1|0.0|.txt', 'EFD|0.05|0.1|0.6|1.txt', 'EFD|0.0|0.1|1.0|0.05|0.05|Ada.txt']
# plot_list = ['EFD|0.05|0.1|0.6|fixed_gamma|.txt', 'EFD|0.05|0.1|1.0|every_round|.txt', 'EFD|0.05|0.1|1.0|break_only|.txt', 'EFD|0.05|0.1|1.0|0.5|Ada.txt', 'EFD|0.05|0.1|0.6|Fix_ada|.txt']
# plot_list = ['EFD|0.05|0.1|0.6|fixed_gamma|.txt', 'EFD|0.05|0.1|1.0|0.5|Ada.txt']
# plot_list = ['EFD|0.05|0.2|0.7|fixed_gamma|.txt', 'EFD|0.05|0.2|1.0|0.5|Ada.txt']
# plot_list = ['EFD|0.05|4|1.0|.txt', 'EFD|0.05|4|1.0|0.5|Ada.txt']
# plot_list = ['EFD|0.05|6|1.0|.txt', 'EFD|0.05|6|0.5|Ada.txt']
# plot_list = ['CHOCO|0.0|0.1|0.0|0.2|.txt']
# plot_list = ['EFD|0.0|0.1|1.0|0.5|.txt', 'EFD|0.0|0.1|0.6|.txt']
# plot_list = ['EFD|0.05|0.05|1.0|0.5|.txt', 'EFD|0.05|0.05|0.6|.txt']
# plot_list = ['EFD|0.01|0.1|0.6|fixed|.txt', 'EFD|0.01|0.1|1.0|0.05|0.4|.txt', 'EFD|0.01|0.1|1.0|0.05|0.5|.txt',
#              'EFD|0.01|0.1|1.0|0.05|0.6|.txt', 'EFD|0.01|0.1|1.0|0.05|0.7|.txt', 'EFD|0.01|0.1|1.0|0.05|0.8|.txt',
#              'EFD|0.01|0.1|1.0|0.05|0.9|.txt']
# plot_list = ['EFD|0.05|0.1|0.6|0.056|0.05|1|20|4|Fix_gamma|.txt', 'EFD|0.05|0.1|1.0|0.056|0.05|1.0|20|4|Ada_pre_0.9|.txt', 'EFD|0.05|0.1|1.0|0.056|0.05|1.0|20|4|Ada_pre_0.99|.txt',
#              'EFD|0.05|0.1|1.0|0.056|0.05|1.0|20|4|Ada_post_0.05|.txt', 'EFD|0.05|0.1|1.0|0.056|0.05|1.0|20|4|Ada_post_0.9|.txt', 'EFD|0.05|0.1|1.0|0.056|0.05|1.0|20|4|Ada_post_0.99|.txt',
#              'EFD|0.05|0.1|1.0|0.056|0.05|1.0|20|4|Ada_post_breakonly|.txt']

# plot_list = ['EFD|0.05|0.1|0.6|1|20|4|Fix_gamma|.txt', 'EFD|0.05|0.1|1.0|20|4|Ada_post_0.9|.txt',
#              'EFD|0.05|0.1|1.0|Ada_post_0.8|.txt']
#
# plot_list = ['EFD|0.05|0.1|0.6|1|20|4|Fix_gamma|.txt', 'EFD|0.05|0.1|1.0|Ada_post_0.98|.txt', 'EFD|0.05|0.1|1.0|Ada_post_0.97|.txt',
#              'EFD|0.05|0.1|1.0|Ada_post_0.8|.txt', 'EFD|0.05|0.1|1.0|20|4|Ada_post_0.9|.txt', 'EFD|0.05|0.1|1.0|Ada_post_0.95|.txt',
#              'EFD|0.05|0.1|1.0|Ada_post_0.94|.txt', 'EFD|0.05|0.1|1.0|Ada_post_0.93|.txt', 'EFD|0.05|0.1|1.0|Ada_post_0.92|.txt', 'EFD|0.05|0.1|1.0|Ada_post_0.91|.txt']

# plot_list = ['EFD|0.05|0.1|0.6|1|20|4|Fix_gamma|.txt', 'EFD|0.05|0.1|0.6|Ada_increasing|.txt', 'EFD|0.05|0.1|0.6|Ada_decreasing|.txt', 'EFD|0.05|0.1|0.01|Ada_0_to_0.1|.txt']
# plot_list = ['EFD|0.05|0.1|0.6|1|20|4|Fix_gamma|.txt', 'EFD|0.05|0.1|1.0|20|4|Ada_pre_0.9|.txt']
# plot_list = ['EFD|0.05|0.1|0.6|1|20|4|Fix_gamma|.txt', 'EFD|0.05|0.1|1.0|Ada_error_norm|.txt']
# plot_list = ['EFD|0.05|0.1|0.6|1|20|4|Fix_gamma|.txt', 'EFD|0.05|0.1|1.0|20|4|Ada_post_0.9|.txt', 'EFD|0.05|0.1|1.0|Ada_post_0.8|.txt']
# plot_list = ['EFD|0.05|0.1|0.6|1|20|4|Fix_gamma|.txt', 'EFD|0.05|0.1|1.0|20|4|Ada_post_0.9|.txt', 'EFD|0.05|0.1|1.0|20|4|Ada_pre_0.9|.txt',  'EFD|0.05|0.1|0.6|Ada_increasing|.txt', 'EFD|0.05|0.1|1.0|Ada_one_e|.txt']
# plot_list = ['EFD|0.05|0.1|0.6|1|20|4|Fix_gamma|.txt', 'EFD|0.05|0.1|1.0|Ada_post_0.98|.txt', 'EFD|0.05|0.1|1.0|Ada_post_0.97|.txt',
#              'EFD|0.05|0.1|1.0|Ada_post_0.8|.txt', 'EFD|0.05|0.1|1.0|20|4|Ada_post_0.9|.txt', 'EFD|0.05|0.1|1.0|Ada_post_0.95|.txt',
#              'EFD|0.05|0.1|1.0|Ada_post_0.94|.txt', 'EFD|0.05|0.1|1.0|Ada_post_0.93|.txt', 'EFD|0.05|0.1|1.0|Ada_post_0.92|.txt', 'EFD|0.05|0.1|1.0|Ada_post_0.91|.txt']

# plot_list = ['EFD|0.05|0.1|0.6|1|20|4|Fix_gamma|.txt', 'EFD|0.05|0.1|1.0|20|4|Ada_post_0.9|.txt', 'EFD|0.05|0.1|1.0|Ada_post_0.89|.txt', 'EFD|0.05|0.1|1.0|Ada_post_0.88|.txt',
#              'EFD|0.05|0.1|1.0|Ada_post_0.87|.txt', 'EFD|0.05|0.1|1.0|Ada_post_0.86|.txt']

# plot_list = ['DCD|0.05|0.2|0.0|CIFAR10|Ring.txt', 'CHOCO|0.05|0.2|0.0|CIFAR10|Ring.txt', 'EFD|0.05|0.2|0.6|CIFAR10|0.056|0.2|Ring.txt']

# plot_list = ['EFD|0.05|0.1|0.6|1|20|4|Fix_gamma|.txt', 'EFD|0.05|0.1|1.0|1|.txt', 'EFD|0.05|0.1|1.0|2|.txt', 'EFD|0.05|0.1|1.0|3|.txt']
# plot_list = ['EFD|0.05|0.2|0.7|Fixed|.txt', 'EFD|0.05|0.2|1.0|Ada|.txt']

# plot_list = ['EFD|0.05|0.05|0.6|Fixed|.txt', 'EFD|0.05|0.05|1.0|Ada|.txt']
# plot_list = ['EFD|0.05|4|1.0|Fixed|1.txt', 'EFD|0.05|4|131.0|Ada|1.txt']

# plot_list = ['EFD|0.05|0.1|1.0|3|.txt', 'EFD|0.05|0.1|1.0|Ada_eta|.txt', 'EFD|0.05|0.1|1.0|Ada_eta_sigma|.txt']

# plot_list = ['EFD|0.05|0.2|0.7|20|5|Fixed|.txt', 'EFD|0.05|0.2|1.0|20|5|Ada|.txt']
# plot_list = ['EFD|0.05|0.1|0.6|20|5|Fixed|.txt', 'EFD|0.05|0.1|1.0|20|5|Ada|.txt']

# plot_list = ['EFD|0.05|0.2|0.7|20|3|Fixed|.txt', 'EFD|0.05|0.2|1.0|20|3|Ada|.txt']
# plot_list = ['EFD|0.05|0.1|0.6|20|3|Fixed|.txt', 'EFD|0.05|0.1|1.0|20|3|Ada|.txt']

# plot_list = ['EFD|0.05|0.2|0.7|20|2|Fixed|.txt', 'EFD|0.05|0.2|1.0|20|2|Ada|.txt']
# plot_list = ['EFD|0.05|0.1|0.6|20|2|Fixed|.txt', 'EFD|0.05|0.1|1.0|20|2|Ada|.txt']

# plot_list = ['EFD|0.5|0.2|0.7|20|5|Fixed|.txt', 'EFD|0.5|0.2|1.0|20|5|Ada|.txt']
# plot_list = ['EFD|0.5|0.1|0.6|20|5|Fixed|.txt', 'EFD|0.5|0.1|1.0|20|5|Ada|.txt']

"alpha=0.01, learning rate might not be the optimal learning rate"
# plot_list = ['EFD|0.01|0.2|0.7|20|5|Fixed|.txt', 'EFD|0.01|0.2|1.0|20|5|Ada|.txt']
# plot_list = ['EFD|0.01|0.1|0.6|20|5|Fixed|.txt', 'EFD|0.01|0.1|1.0|20|5|Ada|.txt']

# plot_list = ['EFD|0.05|0.1|0.6|Fix_ada|.txt', 'DCD|0.05|0.1|0.0|0.056.txt', 'CHOCO|0.05|0.1|0.0|0.056|0.2|.txt', 'BEER|0.05|0.1|0.1|0.1.txt', 'BEER|0.05|0.1|0.2|0.056.txt', 'EFD|0.05|0.1|1.0|Ada_eta_sigma|.txt']
# plot_list = ['EFD|0.0|0.1|1.0|0.1.txt', 'EFD|0.0|0.1|1.0|0.056.txt', 'BEER|0.0|0.1|0.2|0.056.txt', 'BEER|0.0|0.1|0.2|0.1.txt']
# plot_list = ['EFD|0.02|0.1|0.6|0.056|.txt', 'BEER|0.02|0.1|0.2|0.056|.txt']
# plot_list = ['EFD|0.02|0.2|0.6|0.056|.txt', 'BEER|0.02|0.2|0.2|0.056|.txt']

# plot_list = ['BEER|0.05|0.1|0.0316|0.0316|0.05|20|4|.txt', 'CHOCO|0.05|0.1|0.0|0.056|0.2|.txt', 'EFD|0.05|0.1|0.6|0.056.txt']
# plot_list = ['BEER|0.05|0.2|0.0316|0.0316|0.05|20|4|.txt']
# plot_list = ['BEER|0.05|4|0.0316|0.0316|0.05|20|4|.txt']
# plot_list = ['BEER|0.05|6|0.316|0.056|0.05|20|4|.txt']

# plot_list = ['EFD|0.05|6|1.0|0.056|0.05|20|4|Ada.txt', 'EFD|0.05|6|1.0|0.056.txt']
# plot_list = ['EFD|0.05|6|3.0|0.056|0.05|20|4|Ada1.txt', 'EFD|0.05|6|1.0|0.056.txt']

# plot_list = ['EFD|0.05|4|1.0|0.0316|0.05|20|4|Ada.txt', 'EFD|0.05|4|1.0|0.056.txt']
# plot_list = ['EFD|0.05|4|4.0|0.0316|0.05|20|4|Ada1.txt', 'EFD|0.05|4|1.0|0.056.txt']

# plot_list = ['EFD|0.5|0.1|0.1|0.056|0.05|20|2|Ada.txt', 'EFD|0.5|0.1|0.1|0.056|0.05|20|2|.txt']
# plot_list = ['EFD|0.5|0.2|0.1|0.056|0.05|20|2|Ada.txt', 'EFD|0.5|0.2|0.1|0.056|0.05|20|2|.txt']

# plot_list = ['BEER|0.05|0.1|0.2|0.056|20|4|.txt', 'BEER|0.05|0.1|0.3|0.056|20|4|.txt']

# plot_list = ['BEER|0.05|0.1|0.25|0.056|20|4|.txt', 'EFD|0.05|0.1|1.0|Ada|0.056|20|4|.txt',
#              'CHOCO|0.05|0.1|1.0|0.056|0.2|20|4|.txt', 'DCD|0.05|0.1|0.0|0.056|20|4|.txt',
#               'MOTEF|0.05|0.1|0.2|0.01|20|4|M_gradient_start|batch_128|.txt', 'MOTEF|0.05|0.1|0.2|0.01|20|4|M_gradient_start|16|.txt',
#              'MOTEF|0.05|0.1|0.3|0.01|20|4|M_gradient_start|16|.txt', 'MOTEF|0.05|0.1|0.2|0.02|20|4|.txt']

# plot_list = ['BEER|0.05|0.1|0.25|0.056|20|4|.txt', 'EFD|0.05|0.1|1.0|Ada|0.056|20|4|.txt',
#              'CHOCO|0.05|0.1|1.0|0.056|0.2|20|4|.txt', 'DCD|0.05|0.1|0.0|0.056|20|4|.txt',
#              'MOTEF|0.05|0.1|0.2|0.02|20|4|0.05|.txt', 'CEDAS|0.05|0.1|1.0|0.02|20|4|.txt',
#              'DeCoM|0.05|0.1|0.3|0.03|20|4|.txt']

plot_list = ['BEER|0.05|0.2|0.4|0.056|20|4|.txt', 'EFD|0.05|0.2|1.0|Ada|0.056|20|4|.txt',
            'CHOCO|0.05|0.2|0.4|0.056|20|4|.txt', 'DCD|0.05|0.2|0.0|0.056|20|4|.txt',
             'MOTEF|0.05|0.2|0.2|0.02|20|4|.txt',  'CEDAS|0.05|0.2|1.0|0.02|20|4|.txt',
             'DeCoM|0.05|0.2|0.3|0.05|20|4|.txt']

# plot_list = ['BEER|0.05|4|0.3|0.056|20|4|.txt', 'EFD|0.05|4|2.0|Ada|0.056|20|4|.txt',
#              'CHOCO|0.05|4|0.2|0.056|20|4|.txt', 'DCD|0.05|4|0.0|0.056|20|4|.txt',
#              'MOTEF|0.05|4|0.2|0.02|20|4|.txt', 'CEDAS|0.05|4|1.0|0.01|20|4|.txt']
#
# plot_list = ['BEER|0.05|6|0.3|0.056|20|4|.txt', 'EFD|0.05|6|2.45|Ada|0.056|20|4|.txt',
#              'CHOCO|0.05|6|0.2|0.056|20|4|.txt', 'DCD|0.05|6|2.45|0.056|20|4|.txt',
#              'MOTEF|0.05|6|0.2|0.02|20|4|.txt', 'CEDAS|0.05|6|1.0|0.01|20|4|.txt']
#
# plot_list = ['DeCoM|0.05|0.1|0.3|MNIST|0.05|20|4|.txt', 'DeCoM|0.05|8|0.5|MNIST|0.03|20|4|.txt']

color_list = ['blue', 'orange', 'green', 'red', 'gray', 'purple', 'brown', 'pink', 'yellow', 'cyan', 'olive', 'black']

# times = 1
times = 4
# compare = True
compare = False

# Fair = 1
Fair = 0

# agg = 500
agg = 1000
# agg = 20000

iteration = range(agg)
if agg == 1000:
    dataset = "fashion"
else:
    dataset = "CIFAR"

missions = ['acc', 'loss']
# missions = ['loss']
# missions = ['acc', 'loss', 'norm']
# missions = ['norm']

if dataset == 'fashion':
    window_size = 10
else:
    window_size = 250

# print(agg, dataset, window_size)
# plt.subplots(figsize=(10, 4))plt.plot(iteration, x_means, color=color_list[i], label='{}: dc={}'.format(name, dc))
for mission in missions:
    index = int(missions.index(mission)) + 1
    x_means_max = 0
    for i in range(len(plot_list)):
        # print(plot_list[i])
        name = plot_list[i].split('|')[0]
        alpha = plot_list[i].split('|')[1]
        dc = plot_list[i].split('|')[-2]
        compression = plot_list[i].split('|')[2]
        # print(name, dc)
        # if name == 'BEER':
        #     agg = 500
        # elif name == 'EFD' or 'EFDwd':
        #     agg = 1000

        x = pd.read_csv(plot_list[i], header=None)

        x_acc, x_loss = x.values
        x_acc, x_loss = [x_acc[i * agg: (i + 1) * agg] for i in range(times)], [x_loss[i * agg: (i + 1) * agg] for i in range(times)]

        if mission == 'acc':
            x_area = np.stack(x_acc)
        elif mission == 'loss':
            x_area = np.stack(x_loss)

        x_area, x_means = moving_average(input_data=x_area, window_size=window_size)

        x_means = x_area.mean(axis=0)
        x_stds = x_area.std(axis=0, ddof=1)

        # if mission == 'acc':
        #     print(name, dc, x_means[-1], x_stds[-1], '\n')

        if name == 'CHOCO':
            name = 'CHOCO-PSGD'
            if compare:
                plt.plot(iteration, x_means, color=color_list[i], label=r"{}: $\gamma'={}$".format(name, dc))  # dc is consensus in CHOCO case
            else:
                plt.plot(iteration, x_means, color=color_list[i], label='{}'.format(name))
            plt.fill_between(iteration, x_means + x_stds, x_means - x_stds, alpha=0.05, color=color_list[i])
        # elif name == 'AdaG':
        #     name = 'AdaG-PSGD'
        #     plt.plot(iteration, x_means, color=color_list[i], label='{}'.format(name))
        #     plt.fill_between(iteration, x_means + x_stds, x_means - x_stds, alpha=0.05, color=color_list[i])
        # elif name == 'QSADDLe':
        #     name = 'Comp QSADDLe'
        #     plt.plot(iteration, x_means, color=color_list[i], label='{}'.format(name))
        #     plt.fill_between(iteration, x_means + x_stds, x_means - x_stds, alpha=0.05, color=color_list[i])
        elif name == 'DCD':
            name = 'DCD-PSGD'
            plt.plot(iteration, x_means, color=color_list[i], label='{}'.format(name))
            plt.fill_between(iteration, x_means + x_stds, x_means - x_stds, alpha=0.05, color=color_list[i])
        elif name == 'CEDAS':
            name = 'CEDAS'
            plt.plot(iteration, x_means, color=color_list[i], label='{}'.format(name))
            plt.fill_between(iteration, x_means + x_stds, x_means - x_stds, alpha=0.05, color=color_list[i])
        elif name == 'BEER':
            if Fair == 1:
                name = 'BEER (500)'
                x_means = x_means[:500]
                x_stds = x_stds[:500]
                x_b = np.arange(0, 1000, 2)
                plt.plot(x_b, x_means, color=color_list[i], label='{}'.format(name))
                plt.fill_between(x_b, x_means + x_stds, x_means - x_stds, alpha=0.05, color=color_list[i])
            elif Fair == 0:
                name = 'BEER'
                plt.plot(iteration, x_means, color=color_list[i], label='{}'.format(name))
                plt.fill_between(iteration, x_means + x_stds, x_means - x_stds, alpha=0.05, color=color_list[i])
        elif name == 'DeCoM':
            if Fair == 1:
                name = 'DeCoM (500)'
                x_means = x_means[:500]
                x_stds = x_stds[:500]
                x_b = np.arange(0, 1000, 2)
                plt.plot(x_b, x_means, color=color_list[i], label='{}'.format(name))
                plt.fill_between(x_b, x_means + x_stds, x_means - x_stds, alpha=0.05, color=color_list[i])
            elif Fair == 0:
                # if color_list[i] == color_list[0]:
                #     name = 'DeCoM topk (0.1)'
                # elif color_list[i] == color_list[1]:
                #     name = 'DeCoM quantize (8)'
                name = 'DeCoM'
                plt.plot(iteration, x_means, color=color_list[i], label='{}'.format(name))
                plt.fill_between(iteration, x_means + x_stds, x_means - x_stds, alpha=0.05, color=color_list[i])
        elif name == 'MOTEF':
            if Fair == 1:
                name = 'MOTEF (500)'
                x_means = x_means[:500]
                x_stds = x_stds[:500]
                x_b = np.arange(0, 1000, 2)
                plt.plot(x_b, x_means, color=color_list[i], label='{}'.format(name))
                plt.fill_between(x_b, x_means + x_stds, x_means - x_stds, alpha=0.05, color=color_list[i])
            elif Fair == 0:
                name = 'MOTEF'
                plt.plot(iteration, x_means, color=color_list[i], label='{}'.format(name))
                plt.fill_between(iteration, x_means + x_stds, x_means - x_stds, alpha=0.05, color=color_list[i])
        elif name == 'EFDwd' or 'EFD':
            if dc == '1.0':
                compare = False
                name = 'DCD-PSGD+Error Feedback'
            elif dc == '0.0':
                compare = False
                name = 'DCD-PSGD'
            else:
                # compare = True
                name = 'DEFD-PSGD (ours)'
            # print(compare, name)
            if compare:
                plt.plot(iteration, x_means, color=color_list[i], label=r'{}: $\gamma$={}'.format(name, dc))
            else:
                plt.plot(iteration, x_means, color=color_list[i], label='{}'.format(name))
            plt.fill_between(iteration, x_means + x_stds, x_means - x_stds, alpha=0.05, color=color_list[i])

    # plt.title('Quantization \u03B1 = 0')
    # plt.title('Top-k \u03B1 = 0.1')
    # plt.title('Top-k with control \u03B1 = 0')
    # plt.title('Quantization with Control \u03B1 = 0.1')
    plt.xlabel('Aggregations', fontsize=14)
    plt.tick_params(axis='x', which='major', labelsize=14)
    plt.tick_params(axis='y', which='major', labelsize=12)

    if mission == 'acc':
        plt.ylabel('Test Accuracy', fontsize=14)
        if dataset == 'CIFAR':
            plt.ylim([0.4, 0.72])
            # plt.ylim([0.4, 0.80])
        else:
            # pass
            # plt.ylim([0.78, 0.90])
            # plt.ylim([0.65, 0.85])
            # plt.ylim([0.7, 0.82])
            plt.ylim([0.65, 0.82])
        plt.legend(fontsize=14)
        if agg == 1000:
            plt.savefig('{}_{}_{}_{}_{}_{}.pdf'.format(mission, alpha, compression, dc, agg, time.strftime("%H:%M:%S", time.localtime())), bbox_inches='tight')
        elif agg == 20000:
            plt.savefig('{}_{}_{}_{}_{}_{}.png'.format(mission, alpha, compression, dc, agg, time.strftime("%H:%M:%S", time.localtime())), bbox_inches='tight')
        plt.show()

    elif mission == 'loss':
        plt.ylabel('Global Loss', fontsize=14)
        if dataset == 'CIFAR':
            # pass
            plt.ylim([0.005, 0.014])
            # plt.ylim([0.003, 0.010])
        else:
            # pass
            # plt.ylim([0.8, 1.3])
            # plt.ylim([0.0045, 0.0065])
            plt.ylim([0.002, 0.006])
            # plt.ylim([0.002, 0.007])
            # plt.ylim([0.003, 0.013])
        plt.legend(fontsize=14)
        if agg == 1000:
            fig = plt.gcf()
            fig.set_size_inches(6, 4)
            fig.savefig('{}_{}_{}_{}_{}_{}.pdf'.format(mission, alpha, compression, dc, agg, time.strftime("%H:%M:%S", time.localtime())), bbox_inches='tight')
        elif agg == 20000:
            plt.savefig('{}_{}_{}_{}_{}_{}.png'.format(mission, alpha, compression, dc, agg, time.strftime("%H:%M:%S", time.localtime())), bbox_inches='tight')
        plt.show()

# plt.savefig('{}.pdf'.format(mission))
# plt.show()
