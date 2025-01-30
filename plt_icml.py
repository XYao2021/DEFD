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
    return np.array(moving_average), moving_average_means


"20-4 0.05"
"Final plots"
plot_list = [['EFD|0.05|0.2|1.0|Ada|0.056|20|4|.txt', 'BEER|0.05|0.2|0.4|0.056|20|4|.txt',
            'CHOCO|0.05|0.2|0.4|0.056|20|4|.txt', 'DCD|0.05|0.2|0.0|0.056|20|4|.txt',
             'MOTEF|0.05|0.2|0.2|0.02|20|4|.txt',  'CEDAS|0.05|0.2|1.0|0.02|20|4|.txt',
             'DeCoM|0.05|0.2|0.4|0.01|20|4|.txt'], ['EFD|0.05|6|2.45|Ada|0.056|20|4|.txt', 'BEER|0.05|6|0.3|0.056|20|4|.txt',
             'CHOCO|0.05|6|0.2|0.056|20|4|.txt', 'DCD|0.05|6|2.45|0.056|20|4|.txt',
             'MOTEF|0.05|6|0.2|0.02|20|4|.txt', 'CEDAS|0.05|6|1.0|0.01|20|4|.txt', 'DeCoM|0.05|6|0.5|0.02|20|4|.txt']]

"20-2 0.05"

# plot_list = [['EFD|0.05|0.2|Ada|20|2|0.056|.txt', 'BEER|0.05|0.2|0.5|20|2|0.056|.txt', 'CHOCO|0.05|0.2|0.2|20|2|0.056|.txt', 'DCD|0.05|0.2|0|20|2|0.056|.txt',
#              'MOTEF|0.05|0.2|0.5|0.05|20|2|0.02|.txt', 'CEDAS|0.05|0.2|0.2|0.03|20|2|0.01|.txt', 'DeCoM|0.05|0.2|0.3|0.5|20|2|0.02|.txt'],
#              ['EFD|0.05|6|Ada|20|2|0.056|need_redo|.txt', 'BEER|0.05|6|0.5|20|2|0.056|.txt',
#               'CHOCO|0.05|6|0.2|20|2|0.056|.txt', 'DCD|0.05|6|0|20|2|0.056|.txt',
#             'MOTEF|0.05|6|0.5|0.05|20|2|0.02|.txt', 'CEDAS|0.05|6|0.2|0.03|20|2|0.01|.txt', 'DeCoM|0.05|6|0.3|0.001|20|2|0.02|.txt']]

"20-4 CIFAR10"

# plot_list = [['EFD|0.05|0.1|0.6|CIFAR4.txt', 'BEER|0.05|0.1|0.3|CIFAR10|0.056|20|4|.txt', 'CHOCO|0.05|0.1|0.0|CIFAR10.txt', 'DCD|0.05|0.1|0.0|CIFAR4.txt',
#              'MOTEF|0.05|0.1|0.2|CIFAR10|0.05|20|4|0.01|.txt'],
#              ['EFD|0.05|6|1.0|CIFAR10.txt', 'BEER|0.05|6|0.4|CIFAR10|0.056|20|4|.txt', 'CHOCO|0.05|6|0.0|CIFAR10.txt', 'DCD|0.05|6|0.0|CIFAR10.txt',
#              'MOTEF|0.05|6|0.5|CIFAR10|0.005|20|4|0.01|.txt']]

"20-3 0.05"
#
# plot_list = [['EFD|0.05|0.2|Ada|20|3|0.056|.txt', 'BEER|0.05|0.2|0.5|20|3|0.056|.txt', 'CHOCO|0.05|0.2|0.2|20|3|0.056|.txt', 'DCD|0.05|0.2|20|3|0.056|.txt',
#             'MOTEF|0.05|0.2|0.5|0.005|20|3|0.001|.txt', 'CEDAS|0.05|0.2|0.5|0.005|20|3|0.0316|.txt'],
#               ['EFD|0.05|6|Ada|20|3|0.1|.txt', 'BEER|0.05|6|0.3|20|3|0.056|.txt', 'CHOCO|0.05|6|0.2|20|3|0.056|.txt',
#                'DCD|0.05|6|20|3|0.1|.txt', 'MOTEF|0.05|6|0.2|0.005|20|3|0.001|.txt', 'CEDAS|0.05|6|0.5|0.005|20|3|0.0316|.txt']]

color_list = ['blue', 'orange', 'green', 'red', 'gray', 'purple', 'brown', 'pink', 'yellow', 'cyan', 'olive', 'black']

# times = 1
# times = 3
times = 4

Fair = 1
# Fair = 0

agg = 1000
# agg = 20000

iteration = range(agg)
if agg == 1000:
    dataset = "fashion"
elif agg == 20000:
    dataset = "CIFAR"

missions = ['acc', 'loss']

if dataset == 'fashion':
    window_size = 10
else:
    window_size = 250

fig, axs = plt.subplots(2, 2)

for j in range(len(plot_list)):
    for mission in missions:
        index = int(missions.index(mission))
        x_means_max = 0
        folder = plot_list[j]
        for i in range(len(folder)):
            file = folder[i]
            name = file.split('|')[0]
            alpha = file.split('|')[1]
            compression = file.split('|')[2]

            if compression == '4':
                method = 'Quantization'
            elif compression == '6':
                method = 'Quantization'
            elif compression == '0.2':
                method = 'Top-k'
            elif compression == '0.1':
                method = 'Top-k'

            x = pd.read_csv(file, header=None)

            x_acc, x_loss = x.values

            x_acc, x_loss = [x_acc[i * agg: (i + 1) * agg] for i in range(times)], [x_loss[i * agg: (i + 1) * agg] for i in range(times)]
            if len(x_acc) > times:
                x_acc, x_loss = x_acc[1:], x_loss[1:]

            if mission == 'acc':
                x_area = np.stack(x_acc)
            elif mission == 'loss':
                x_area = np.stack(x_loss)

            x_area, x_means = moving_average(input_data=x_area, window_size=window_size)

            x_means = x_area.mean(axis=0)
            x_stds = x_area.std(axis=0, ddof=1)

            axs[index][j].rcParams['pdf.fonttype'] = 42
            axs[index][j].rcParams['ps.fonttype'] = 42
            axs[index][j].rcParams['font.family'] = 'arial'

            if name == 'CHOCO':
                name = 'CHOCO'
                axs[index][j].plot(iteration, x_means, color=color_list[i], label='{}'.format(name))
                axs[index][j].fill_between(iteration, x_means + x_stds, x_means - x_stds, alpha=0.05, color=color_list[i])
            elif name == 'DCD':
                name = 'DCD'
                axs[index][j].plot(iteration, x_means, color=color_list[i], label='{}'.format(name))
                axs[index][j].fill_between(iteration, x_means + x_stds, x_means - x_stds, alpha=0.05, color=color_list[i])
            elif name == 'CEDAS':
                name = 'CEDAS'
                axs[index][j].plot(iteration, x_means, color=color_list[i], label='{}'.format(name))
                axs[index][j].fill_between(iteration, x_means + x_stds, x_means - x_stds, alpha=0.05, color=color_list[i])
            elif name == 'BEER':
                if Fair == 1:
                    name = 'BEER'
                    y = int(len(x_means))
                    x_means = x_means[:int(y / 2)]
                    x_stds = x_stds[:int(y / 2)]
                    x_b = np.arange(0, y, 2)
                    axs[index][j].plot(x_b, x_means, color=color_list[i], label='{}'.format(name))
                    axs[index][j].fill_between(x_b, x_means + x_stds, x_means - x_stds, alpha=0.05, color=color_list[i])
                elif Fair == 0:
                    name = 'BEER'
                    axs[index][j].plot(iteration, x_means, color=color_list[i], label='{}'.format(name))
                    axs[index][j].fill_between(iteration, x_means + x_stds, x_means - x_stds, alpha=0.05, color=color_list[i])
            elif name == 'DeCoM':
                if Fair == 1:
                    name = 'DeCoM'
                    x_means = x_means[:int(agg / 2)]
                    x_stds = x_stds[:int(agg / 2)]
                    x_b = np.arange(0, agg, 2)
                    axs[index][j].plot(x_b, x_means, color=color_list[i], label='{}'.format(name))
                    axs[index][j].fill_between(x_b, x_means + x_stds, x_means - x_stds, alpha=0.05, color=color_list[i])
                elif Fair == 0:
                    name = 'DeCoM'
                    axs[index][j].plot(iteration, x_means, color=color_list[i], label='{}'.format(name))
                    axs[index][j].fill_between(iteration, x_means + x_stds, x_means - x_stds, alpha=0.05, color=color_list[i])
            elif name == 'MOTEF':
                if Fair == 1:
                    name = 'MOTEF'
                    x_means = x_means[:int(agg / 2)]
                    x_stds = x_stds[:int(agg / 2)]
                    x_b = np.arange(0, agg, 2)
                    axs[index][j].plot(x_b, x_means, color=color_list[i], label='{}'.format(name))
                    axs[index][j].fill_between(x_b, x_means + x_stds, x_means - x_stds, alpha=0.05, color=color_list[i])
                elif Fair == 0:
                    name = 'MOTEF'
                    axs[index][j].plot(iteration, x_means, color=color_list[i], label='{}'.format(name))
                    axs[index][j].fill_between(iteration, x_means + x_stds, x_means - x_stds, alpha=0.05, color=color_list[i])
            elif name == 'EFD':
                name = 'DEFD (ours)'
                axs[index][j].plot(iteration, x_means, color=color_list[i], label='{}'.format(name))
                axs[index][j].fill_between(iteration, x_means + x_stds, x_means - x_stds, alpha=0.05, color=color_list[i])

        if method == 'Top-k':
            axs[0][j].set_title('{} (k={}) \u03B1 = {}'.format(method, compression, alpha))
        elif method == 'Quantization':
            axs[0][j].set_title('{} ({}-bits) \u03B1 = {}'.format(method, compression, alpha))
        if mission == 'acc':
            if dataset == 'CIFAR':
                axs[index][j].set_ylim([0.3, 0.72])
            elif dataset == 'fashion':
                axs[index][j].set_ylim([0.65, 0.82])

        elif mission == 'loss':
            if dataset == 'CIFAR':
                axs[index][j].set_ylim([0.005, 0.014])
            elif dataset == 'fashion':
                axs[index][j].set_ylim([0.003, 0.008])
        axs[index][j].grid()
    # print('\n')

# plt.xlabel('Aggregations', fontsize=14)
# plt.tick_params(axis='x', which='major', labelsize=14)
# plt.tick_params(axis='y', which='major', labelsize=12)
axs[1][0].set_xlabel('Aggregations', fontsize=14)
axs[1][1].set_xlabel('Aggregations', fontsize=14)
axs[0][0].set_ylabel('Test Accuracy', fontsize=14)
axs[1][0].set_ylabel('Global Loss', fontsize=14)
axs[0][0].legend(bbox_to_anchor=(0.2, 1.38), loc='upper left', ncol=4, frameon=True, fontsize='small')
# axs[0][0].legend(bbox_to_anchor=(0.03, 1.3), loc='upper left', ncol=5, frameon=True, fontsize='small')
# plt.legend()
# compression = 'Topk'
# compression = 'Quantize'

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

plt.savefig('{}_{}_{}.pdf'.format(method, dataset, alpha))
plt.show()
