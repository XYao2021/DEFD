import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd

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
                # print(i, j, input_data[i][j + 1], type(input_data[i][j + 1]))
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
# plot_list = ['EFD|0.05|8|0.0|CIFAR10.txt', 'EFD|0.05|8|1.0|CIFAR10.txt']
# plot_list = ['CHOCO|0.05|6|0.0|CIFAR10.txt', 'EFD|0.05|6|0.0|CIFAR10.txt', 'EFD|0.05|6|1.0|CIFAR10.txt']
# plot_list = ['CHOCO|0.05|0.1|0.0|CIFAR10.txt', 'AdaG|0.05|0.1|0.0|False|CIFAR10|0.056|0.002|.txt', 'DCD|0.05|0.1|0.0|CIFAR4.txt', 'EFD|0.05|0.1|0.6|CIFAR4.txt']
plot_list = ['CHOCO|0.05|0.2|0.0|CIFAR4.txt', 'AdaG|0.05|0.2|0.0|False|CIFAR10|0.056|0.004|.txt', 'DCD|0.05|0.2|0.0|CIFAR4.txt', 'EFD|0.05|0.2|0.6|CIFAR4.txt']

"FashionMNIST topk"  # AdaG has the problem with consensus for top-k compression
# plot_list = ['CHOCO|0.05|0.1|0.0|.txt', 'AdaG|0.05|0.1|0.0|0.056.txt', 'QSADDLe|0.05|0.1|0.4|.txt', 'DCD|0.05|0.1|0.0|.txt', 'EFD|0.05|0.1|0.6|1.txt']
# plot_list = ['CHOCO|0.05|0.2|0.0|.txt', 'AdaG|0.05|0.2|0.0|0.056.txt', 'QSADDLe|0.05|0.2|0.5|.txt', 'DCD|0.05|0.2|0.0|.txt', 'EFD|0.05|0.2|0.9|1.txt']

"Compare the influence from gamma"
# plot_list = ['EFD|0.05|0.1|0.1|0.056.txt', 'EFD|0.05|0.1|0.2|0.056.txt', 'EFD|0.05|0.1|0.3|0.056.txt', 'EFD|0.05|0.1|0.4|0.056.txt',
#              'EFD|0.05|0.1|0.5|0.056.txt', 'EFD|0.05|0.1|0.6|0.056.txt', 'EFD|0.05|0.1|0.7|0.056.txt', 'EFD|0.05|0.1|0.8|0.056.txt',
#              'EFD|0.05|0.1|0.9|0.056.txt', 'EFD|0.05|0.1|1.0|0.056.txt']

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
# plot_list = ['CHOCO|0.01|4|0.0|0.056|0.2|.txt', 'EFD|0.01|4|0.0|0.056.txt', 'EFD|0.01|4|1.0|0.056.txt']
# plot_list = ['CHOCO|0.01|6|0.0|0.056|0.2|.txt', 'EFD|0.01|6|0.0|0.0316.txt', 'EFD|0.01|6|1.0|0.056.txt']

color_list = ['blue', 'orange', 'green', 'red', 'purple', 'yellow', 'brown', 'pink', 'gray', 'cyan', 'olive', 'black']

times = 4
# compare = True
compare = False

# agg = 500
# agg = 1000
agg = 20000

iteration = range(agg)
if agg == 1000:
    dataset = "fashion"
elif agg == 500:
    dataset = "fashion"
else:
    dataset = "CIFAR"

missions = ['acc', 'loss']
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
        name = plot_list[i].split('|')[0]
        alpha = plot_list[i].split('|')[1]
        dc = plot_list[i].split('|')[-2]
        compression = plot_list[i].split('|')[2]
        # print(name, dc)

        x = pd.read_csv(plot_list[i], header=None)
        # if name == 'CHOCO':
        #     x_acc, x_loss = x.values
        #     x_acc, x_loss = [x_acc[i * agg: (i + 1) * agg] for i in range(times)], [x_loss[i * agg: (i + 1) * agg]for i in range(times)]
        #
        # elif name == 'EFD':
        #     x_acc, x_loss, x_norm = x.values
        #     x_acc, x_loss, x_norm = [x_acc[i * agg: (i + 1) * agg] for i in range(times)], [x_loss[i * agg: (i + 1) * agg] for i in range(times)], [x_norm[i * agg: (i + 1) * agg] for i in range(times)]

        x_acc, x_loss = x.values
        # print(len(x_acc), len(x_loss))
        # print(x_acc, x_loss)
        x_acc, x_loss = [x_acc[i * agg: (i + 1) * agg] for i in range(times)], [x_loss[i * agg: (i + 1) * agg] for i in range(times)]

        # print(len(x_acc), len(x_loss))
        if mission == 'acc':
            x_area = np.stack(x_acc)
        elif mission == 'loss':
            x_area = np.stack(x_loss)
        # elif mission == 'norm':
        #     x_area = np.stack(x_norm)
        # print(len(x_area[0]))

        x_area, x_means = moving_average(input_data=x_area, window_size=window_size)

        x_means = x_area.mean(axis=0)
        x_stds = x_area.std(axis=0, ddof=1)

        if mission == 'acc':
            print(name, dc, x_means[-1], x_stds[-1], '\n')

        # plt.subplot(1, len(missions), index)
        # print(name)
        # if len(plot_list) == 2:
        #     if name == 'CHOCO':
        #         plt.plot(iteration, x_means, color=color_list[i], label='{}'.format(name))
        #         # plt.plot(iteration, x_means, label='{}'.format(name))
        #     else:
        #         plt.plot(iteration, x_means, color=color_list[i], label='{}: dc={}'.format(name, dc))
        #         # plt.plot(iteration, x_means, label='{}: dc={}'.format(name, dc))
        # else:
        #     plt.plot(iteration, x_means, color=color_list[i], label='dc={}'.format(dc))
        #     # plt.plot(iteration, x_means, label='dc={}'.format(dc))
        if name == 'CHOCO':
            # if compare:
            #     plt.plot(iteration, x_means, color=color_list[i], label='{}:{}'.format(name, dc))  # dc is consensus in CHOCO case
            # else:
            #     plt.plot(iteration, x_means, color=color_list[i], label='{}'.format(name))
            plt.plot(iteration, x_means, color=color_list[i], label=r"{}: $\gamma'={}$".format(name, dc))
        elif name == 'AdaG':
            plt.plot(iteration, x_means, color=color_list[i], label='{}'.format(name))
        elif name == 'QSADDLe':
            plt.plot(iteration, x_means, color=color_list[i], label='{}'.format(name))
        elif name == 'DCD':
            plt.plot(iteration, x_means, color=color_list[i], label='{}'.format(name))
        elif name == 'EFDwd' or 'EFD':
            name = 'DEED'
            # if compare:
            #     plt.plot(iteration, x_means, color=color_list[i], label='{}: {}'.format(name, dc))
            # else:
            #     plt.plot(iteration, x_means, color=color_list[i], label='{}'.format(name))
            plt.plot(iteration, x_means, color=color_list[i], label=r'{}: $\gamma$={}'.format(name, dc))

        # print(color_list[i])
        plt.fill_between(iteration, x_means+x_stds, x_means-x_stds, alpha=0.05, color=color_list[i])
        # plt.fill_between(iteration, x_means + x_stds, x_means - x_stds, alpha=0.1)

    # plt.title('Quantization \u03B1 = 0')
    # plt.title('Top-k \u03B1 = 0.1')
    # plt.title('Top-k with control \u03B1 = 0')
    # plt.title('Quantization with Control \u03B1 = 0.1')
    plt.xlabel('Aggregations', fontsize=14)

    if mission == 'acc':
        plt.ylabel('Test Accuracy', fontsize=14)
        if dataset == 'CIFAR':
            # plt.ylim([0.4, 0.72])
            plt.ylim([0.4, 0.80])
        else:
            # plt.ylim([0.55, 0.75])
            # plt.ylim([0.55, 0.81])
            # plt.ylim([0.7, 0.85])
            plt.ylim([0.5, 0.8])
        plt.legend()
        if agg == 1000:
            plt.savefig('{}_{}_{}_{}_{}.pdf'.format(mission, alpha, compression, dc, agg))
        else:
            plt.savefig('{}_{}_{}_{}_{}.png'.format(mission, alpha, compression, dc, agg))
        plt.show()

    elif mission == 'loss':
        plt.ylabel('Global Loss', fontsize=14)
        if dataset == 'CIFAR':
            # plt.ylim([0.005, 0.014])
            plt.ylim([0.003, 0.010])
        else:
            # plt.ylim([0.0045, 0.0065])
            plt.ylim([0.003, 0.008])
            # plt.ylim([0.002, 0.007])
            # plt.ylim([0.005, 0.013])
        plt.legend()
        if agg == 1000:
            plt.savefig('{}_{}_{}_{}_{}.pdf'.format(mission, alpha, compression, dc, agg))
        else:
            plt.savefig('{}_{}_{}_{}_{}.png'.format(mission, alpha, compression, dc, agg))
        plt.show()

    # elif mission == 'norm':
    #     plt.ylabel('value of {}'.format(r'$\gamma$'))
    #     if dataset == "fashion":
    #         if compression == '4':
    #             plt.ylim([0.1, 1])
    #         elif compression == '8':
    #             pass
    #         elif compression == '0.1':
    #             plt.ylim([0, 0.7])
    #         elif compression == '0.2':
    #             plt.ylim([0, 0.5])
    #     elif dataset == "CIFAR":
    #         pass
        # plt.plot(iteration, [0.11430799414888455 for i in range(len(iteration))])
        # plt.plot(iteration, [0.11430799414888455 * 2 for i in range(len(iteration))])
        # plt.plot(iteration, [0.11430799414888455 * 3 for i in range(len(iteration))])
        # plt.plot(iteration, [0.11430799414888455 * 4.5 for i in range(len(iteration))])
        # plt.ylim([0.003, 0.015])
        # plt.legend()
        # plt.savefig('{}_{}_{}_{}_{}.pdf'.format('gamma', alpha, compression, dc, agg))
        # plt.show()

# plt.savefig('{}.pdf'.format(mission))
# plt.show()
