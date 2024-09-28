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
                moving_average[i].append(sum(input_data[i][:j + 1]) / len(input_data[i][:j + 1]))
                # print(i, sum(input_data[:i+1]) / len(input_data[:i+1]))
            else:
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

# plot_list = ['EFD|0.0|0.1|False|1.0|.txt', 'EFD|0.0|0.1|False|0.95|.txt', 'EFD|0.0|0.1|False|0.9|.txt', 'EFD|0.0|0.1|False|0.85|.txt']
plot_list = ['EFDwd|0.2|0.05|.txt', 'CHOCO|0.2|0.05|.txt', 'DCD|0.2|0.05|.txt', 'ECD|0.2|0.05|.txt']

color_list = ['red', 'blue', 'brown', 'aqua']

# plot_list = ['EFD|8|0|.txt', 'CHOCOe|8|0|.txt']
# plot_list = ['EFDwd|0|5|.txt', 'DCD|0|5|.txt']
# color_list = ['red', 'blue']
#
# plot_list = ['EFDwd|CIFAR10|0.2|0.1|.txt']
# color_list = ['red']

times, agg = 4, 500

# times, agg = 4, 5000
iteration = range(agg)

missions = ['acc', 'loss']
# missions = ['acc', 'loss', 'norm']
# missions = ['norm']
window_size = 10

plt.subplots(figsize=(10, 4))
for mission in missions:
    index = int(missions.index(mission)) + 1
    x_means_max = 0
    for i in range(len(plot_list)):
        x = pd.read_csv(plot_list[i], header=None)

        if len(missions) == 2:
            x_acc, x_loss = x.values
            x_acc, x_loss = [x_acc[i * agg: (i + 1) * agg] for i in range(times)], [x_loss[i * agg: (i + 1) * agg]for i in range(times)]

        else:
            x_acc, x_loss, x_norm = x.values
            x_acc, x_loss, x_norm = [x_acc[i * agg: (i + 1) * agg] for i in range(times)], [x_loss[i * agg: (i + 1) * agg] for i in range(times)], [x_norm[i * agg: (i + 1) * agg] for i in range(times)]

        if mission == 'acc':
            x_area = np.stack(x_acc)
        elif mission == 'loss':
            x_area = np.stack(x_loss)
        elif mission == 'norm':
            x_area = np.stack(x_norm)

        x_area, x_means = moving_average(input_data=x_area, window_size=window_size)

        # x_means = x_area.mean(axis=0)
        x_stds = x_area.std(axis=0, ddof=1)

        name = plot_list[i].split('|')[0]
        plt.subplot(1, len(missions), index)
        plt.plot(iteration, x_means, color=color_list[i], label='{}'.format(name))
        plt.fill_between(iteration, x_means+x_stds, x_means-x_stds, alpha=0.1, color=color_list[i])

        if name == 'DCD':
            name = 'DCD-PSGD'
        elif name == 'ECD':
            name = 'ECD-PSGD'
        elif name == 'CHOCO':
            name = 'CHOCO-SGD'
        elif name == 'RCD':
            name = 'EFD-PSGD (old)'
        elif name == 'RCD' or 'RCD_L':
            name = 'EFD-PSGD (ours)'

    # plt.title('Quantization \u03B1 = 0')
    # plt.title('Top-k \u03B1 = 0.1')
    # plt.title('Top-k with control \u03B1 = 0')
    # plt.title('Quantization with Control \u03B1 = 0.1')
    plt.xlabel('Aggregations')

    if mission == 'acc':
        plt.ylabel('Test Accuracy')
        # plt.ylim([0.1, 0.82])
        # plt.ylim([0.4, 0.82])
        plt.legend()
        # plt.savefig('{}.pdf'.format(mission))
        # plt.show()

    elif mission == 'loss':
        plt.ylabel('Global Loss')
        plt.ylim([0.003, 0.015])
        plt.legend()
        # plt.savefig('{}.pdf'.format(mission))
        # plt.show()

    elif mission == 'norm':
        plt.ylabel('alpha value')
        # plt.plot(iteration, [0.11430799414888455 for i in range(len(iteration))])
        # plt.plot(iteration, [0.11430799414888455 * 2 for i in range(len(iteration))])
        # plt.plot(iteration, [0.11430799414888455 * 3 for i in range(len(iteration))])
        # plt.plot(iteration, [0.11430799414888455 * 4.5 for i in range(len(iteration))])
        # plt.ylim([0.003, 0.015])
        plt.legend()
        plt.savefig('{}.pdf'.format(mission))
        plt.show()

plt.savefig('{}.pdf'.format(mission))
plt.show()
