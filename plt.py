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

plot_list = []

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
