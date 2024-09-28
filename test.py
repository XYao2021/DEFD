import random
import h5py
import numpy as np
import sympy
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import math
import torchvision
import pandas as pd
import glob
import os
import sys
from scipy.spatial.distance import squareform
import scipy as sp
import networkx
import torch.distributed as dist
import time
import timeit
from datetime import date
import struct
from tabulate import tabulate
import pandas as pd
# from brokenaxes import brokenaxes
import matplotlib.pyplot as plt
import torch.nn.functional as F
from model.MNISTModel import MNISTModel
from model.CIFAR10Model import CIFAR10Model
from matplotlib import pyplot as plt
import numpy as np
# from cycler import cycler
# import matplotlib.ticker as ticker
from numpy.random import RandomState, SeedSequence
from numpy.random import MT19937
from model.model import *
from sklearn.cluster import KMeans
import itertools
import json
from sympy import Symbol
from sympy.solvers.inequalities import reduce_rational_inequalities
# import matplotlib.gridspec as gridspec
# from matplotlib.patches import Rectangle


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

def matrix(nodes, num_neighbor):
    upper = int(nodes / 2) - 2
    bottom = 1
    # if num_neighbor + 1 > int(nodes / 2) - 1:
    #     print('Number should be in range [{}, {}]'.format(bottom, upper))
    #     raise Exception('Invalid neighbor number')
    matrix = np.ones((nodes,), dtype=int)
    while True:
        org_matrix = np.diag(matrix)
        org_target = np.arange(nodes, dtype=int)
        for i in range(nodes):
            if np.count_nonzero(org_matrix[i]) < num_neighbor + 1:
                if np.count_nonzero(org_matrix[i]) < num_neighbor + 1 and np.count_nonzero(
                        org_matrix.transpose()[i]) < num_neighbor + 1:
                    target = np.setdiff1d(org_target, i)
                    target_set = []
                    for k in range(len(target)):
                        if np.count_nonzero(org_matrix[target[k]]) < num_neighbor + 1:
                            target_set.append(target[k])
                    if num_neighbor + 1 - int(np.count_nonzero(org_matrix[i])) <= len(target_set):
                        target = np.random.choice(target_set, num_neighbor + 1 - int(np.count_nonzero(org_matrix[i])),
                                                  replace=False)
                    for j in range(len(target)):
                        org_matrix[i][target[j]] = 1
                        org_matrix.transpose()[i][target[j]] = 1
            else:
                pass
        if np.count_nonzero(
                np.array([np.count_nonzero(org_matrix[i]) for i in range(nodes)]) - (num_neighbor + 1)) == 0:
            break
    return org_matrix

def Check_Matrix(client, matrix):
    count = 0
    for i in range(client):
        if np.count_nonzero(matrix[i] - matrix.transpose()[i]) == 0:
            pass
        else:
            count += 1
    if count != 0:
        raise Exception('The Transfer Matrix Should be Symmetric')
    else:
        print('Transfer Matrix is Symmetric Matrix')


# plot_list = ['EFD|0.0|8|False|.txt', 'CHOCO|0.0|8|False|.txt', 'DCD|0.0|8|False|.txt', 'ECD|0.0|8|False|.txt']
# color_list = ['red', 'blue', 'green', 'orange']
#
# times, agg = 4, 500
# iteration = range(agg)
# missions = ['acc', 'loss']
# window_size = 10
#
# start = 450
#
# fig = plt.figure(figsize=(10, 4))
# subfigs = fig.subfigures(1, 2, wspace=0.07)
#
# for mission in missions:
#     index = int(missions.index(mission)) + 1
#     main_ax = subfigs[index - 1].subplots(1, 1)
#     small_ax = subfigs[index - 1].add_axes([0.55, 0.4, 0.32, 0.32])  # [left, bottom, width, height]
#     for i in range(len(plot_list)):
#         x = pd.read_csv(plot_list[i], header=None)
#         x_acc, x_loss = x.values
#         x_acc, x_loss = [x_acc[i * agg: (i + 1) * agg] for i in range(times)], [x_loss[i * agg: (i + 1) * agg]for i in range(times)]
#
#         if mission == 'acc':
#             x_area = np.stack(x_acc)
#         elif mission == 'loss':
#             x_area = np.stack(x_loss)
#
#         x_area, x_means = moving_average(input_data=x_area, window_size=window_size)
#         x_stds = x_area.std(axis=0, ddof=1)
#
#         name = plot_list[i].split('|')[0]
#
#         # main_ax = subfigs[index-1].subplots(1, 1)
#         main_ax.plot(iteration, x_means, color=color_list[i], label='{}'.format(name))
#         main_ax.fill_between(iteration, x_means+x_stds, x_means-x_stds, alpha=0.1, color=color_list[i])
#
#         if mission == 'acc':
#             main_ax.set_ylim([0.1, 0.82])
#         elif mission == 'loss':
#             main_ax.set_ylim([0.004, 0.014])
#
#         # Plot something in the small figure
#         small_ax.plot(iteration[start:], x_means[start:], color=color_list[i])
#         small_ax.fill_between(iteration[start:], x_means[start:] + x_stds[start:], x_means[start:] - x_stds[start:], alpha=0.1, color=color_list[i])
#
#         if mission == 'acc':
#             small_ax.set_ylim([0.7, 0.82])
#         elif mission == 'loss':
#             small_ax.set_ylim([0.004, 0.008])
#
#     main_ax.legend(bbox_to_anchor=(-0.1, 1.35), loc='upper center', ncol=4)
#
# plt.legend()
# # plt.savefig('test.pdf')
# plt.show()
# # Create a larger figure
# # Create main figure and axes
# fig, main_ax = plt.subplots()
#
# # Plot something in the main figure
# main_ax.plot(np.linspace(0, 10, 100), np.sin(np.linspace(0, 10, 100)), label='Main Figure')
#
# # Create a small figure inside the main figure
# small_ax = fig.add_axes([0.55, 0.3, 0.32, 0.32])  # [left, bottom, width, height]
#
# # Plot something in the small figure
# small_ax.plot(np.linspace(0, 5, 50), np.cos(np.linspace(0, 5, 50)), label='Small Figure')
#
# # # Set labels for the small figure
# # small_ax.set_xlabel('X-axis (Small)')
# # small_ax.set_ylabel('Y-axis (Small)')
#
# # Set labels for the main figure
# main_ax.set_xlabel('X-axis (Main)')
# main_ax.set_ylabel('Y-axis (Main)')

# plot_list = ['EFD|0.1|0.1|2|False|1.0|quan|.txt', 'EFD|0.1|0.1|4|False|1.0|quan|.txt', 'EFD|0.1|0.1|6|False|1.0|quan|.txt', 'EFD|0.1|0.1|8|False|1.0|quan|.txt']
# plot_list = ['EFD|0.0|0.1|2|False|1.0|quan|.txt', 'EFD|0.0|0.1|4|False|1.0|quan|.txt', 'EFD|0.0|0.1|6|False|1.0|quan|.txt', 'EFD|0.0|0.1|8|False|1.0|quan|.txt']
#
# plot_list = ['EFD|0.0|0.1|False|1.0|.txt', 'EFD|0.0|0.1|False|0.95|.txt', 'EFD|0.0|0.1|False|0.9|.txt', 'EFD|0.0|0.1|False|0.85|.txt', 'EFD|0.0|0.1|False|0.8|.txt',
#              'EFD|0.0|0.1|False|0.7|.txt', 'EFD|0.0|0.1|False|0.6|.txt', 'EFD|0.0|0.1|False|0.5|.txt', 'EFD|0.0|0.1|False|0.5|.txt', 'EFD|0.0|0.1|False|0.3|.txt',
#              'EFD|0.0|0.1|False|0.2|.txt', 'EFD|0.0|0.1|False|0.1|.txt']
# # plot_list = ['EFD|0.0|0.2|False|0.8|.txt', 'EFD|0.0|0.2|False|0.85|.txt', 'EFD|0.0|0.2|False|0.9|.txt', 'EFD|0.0|0.2|False|0.95|.txt', 'EFD|0.0|0.2|False|1.0|.txt']
# # plot_list = ['EFD|0.0|0.3|False|0.8|.txt', 'EFD|0.0|0.3|False|0.85|.txt', 'EFD|0.0|0.3|False|0.9|.txt', 'EFD|0.0|0.3|False|0.95|.txt', 'EFD|0.0|0.3|False|1.0|.txt']
# # plot_list = ['EFD|0.0|0.4|False|0.8|.txt', 'EFD|0.0|0.4|False|0.85|.txt', 'EFD|0.0|0.4|False|0.9|.txt', 'EFD|0.0|0.4|False|0.95|.txt', 'EFD|0.0|0.4|False|1.0|.txt']
#
# # plot_list = ['EFD|0.1|0.2|False|0.8|.txt', 'EFD|0.1|0.2|False|0.85|.txt', 'EFD|0.1|0.2|False|0.9|.txt', 'EFD|0.1|0.2|False|0.95|.txt', 'EFD|0.1|0.2|False|1.0|.txt']
#
# # plot_list = ['EFD|0.0|0.1|False|0.8|1000|.txt', 'EFD|0.0|0.1|False|0.85|1000|.txt', 'EFD|0.0|0.1|False|0.9|1000|.txt', 'EFD|0.0|0.1|False|0.95|1000|.txt', 'EFD|0.0|0.1|False|1.0|1000|.txt']
#
# # color_list = ['red', 'blue', 'green', 'orange', 'aqua']
# color_list = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'aqua', 'black', 'yellow']
#
# # plot_list = ['EFD|0.1|0|test|.txt', 'CHOCO|topk|0.1|0|.txt']
# plot_list = ['EFD|0.1|0.1|test|.txt', 'CHOCO|topk|0.1|0.1|.txt']
# color_list = ['red', 'blue']
# times, agg = 4, 500
#
# iteration = range(agg)
#
# missions = ['acc', 'loss', 'alpha']
#
# window_size = 30
# # plt.subplots(figsize=(10, 4))
# for mission in missions:
#     index = int(missions.index(mission)) + 1
#     x_means_max = 0
#     for i in range(len(plot_list)):
#         name = plot_list[i].split('|')[0]
#         print(i, name)
#         x = pd.read_csv(plot_list[i], header=None)
#         if name == 'EFD':
#             x_acc, x_loss, alpha = x.values
#             x_acc, x_loss, alpha = [x_acc[i * agg: (i + 1) * agg] for i in range(times)], [x_loss[i * agg: (i + 1) * agg]for i in range(times)], [alpha[i * agg: (i + 1) * agg]for i in range(times)]
#         else:
#             x_acc, x_loss = x.values
#             x_acc, x_loss = [x_acc[i * agg: (i + 1) * agg] for i in range(times)], [x_loss[i * agg: (i + 1) * agg]for i in range(times)]
#         # print(alpha)
#         if mission == 'acc':
#             x_area = np.stack(x_acc)
#         elif mission == 'loss':
#             x_area = np.stack(x_loss)
#         elif mission == 'alpha':
#             x_area = np.stack(alpha)
#
#         x_area, x_means = moving_average(input_data=x_area, window_size=window_size)
#
#         # x_means = x_area.mean(axis=0)
#         x_stds = x_area.std(axis=0, ddof=1)
#         # if name == 'quan':
#         #     name = 2*(i+1)
#
#         # plt.subplot(1, 3, index)
#         plt.plot(iteration, x_means, color=color_list[i], label='{}'.format(name))
#         # plt.plot(iteration, x_means, label='{}'.format(name))
#         plt.fill_between(iteration, x_means+x_stds, x_means-x_stds, alpha=0.1, color=color_list[i])
#         # plt.fill_between(iteration, x_means + x_stds, x_means - x_stds, alpha=0.1)
#
#         if mission == 'acc':
#             plt.ylabel('Test Accuracy')
#             plt.ylim([0.4, 0.82])
#             # plt.ylim([0.75, 0.82])
#
#         elif mission == 'loss':
#             plt.ylabel('Global Loss')
#             plt.ylim([0.003, 0.010])
#             # plt.ylim([0.004, 0.005])
#
#         elif mission == 'alpha':
#             plt.ylabel('Alpha')
#             plt.plot(iteration, [0.11430799414888455 for i in range(len(iteration))])
#             plt.plot(iteration, [0.11430799414888455 * 2 for i in range(len(iteration))])
#             # plt.plot(iteration, [0.11430799414888455 * 3 for i in range(len(iteration))])
#             # plt.plot(iteration, [0.11430799414888455 * 4 for i in range(len(iteration))])
#             # plt.ylim([0.1, ])
#
#     plt.legend()
#     plt.savefig('{}.pdf'.format(mission))
#     plt.show()
"""beta upper bound and upper bound comparison with given topology START"""
# Ring/random-20-4/random-20-9/random-20-15/fully-connected
# nodes = 20
# neighbors = 4
# np.random.seed(13)
# conn_matrix = matrix(nodes=nodes, num_neighbor=neighbors)
# # print(conn_matrix)
# conn_matrix = conn_matrix * (1/(neighbors+1))
#
# Check_Matrix(client=nodes, matrix=conn_matrix)
#
# lambdas = np.linalg.eigvalsh(conn_matrix)
# lambdas = sorted(lambdas)[::-1][1:]
#
# mu = max([abs(lambdas[i]-1) for i in range(len(lambdas))])
#
# rho = 0
# for i in range(len(lambdas)):
#     if lambdas[i] == 1.0:
#         pass
#     else:
#         rho = lambdas[i]
#         break
#
# dcs = np.arange(0, 1.05, 0.02)
# dcs = dcs[1:]
# numerator = (1-rho)**2
# denominator_1 = 4*(mu**2)
#
# DCD_upper = numerator / denominator_1
# EFD_upper = []
# new_dcs = []
# for dc in dcs:
#     denominator_2 = (mu*(np.sqrt(2)*dc+1)+np.sqrt(2)*dc*(1-rho))**2
#     EFD_upper_dc = numerator / denominator_2
#     EFD_upper.append(EFD_upper_dc)
#     if EFD_upper_dc >= DCD_upper:
#         new_dcs.append(dc)
# DCD_upper = [DCD_upper for i in range(len(dcs))]
#
# print('EFD beta upper bound: ', EFD_upper, '\n')
# print('DCD beta upper bound: ', DCD_upper[0], '\n')

# plt.plot(dcs, EFD_upper, label='DEED')
# plt.plot(dcs, DCD_upper, label='DCD')
# plt.xlabel(r'value of $\gamma$', fontsize=16)
# plt.ylabel(r'Upper bound of $\beta$', fontsize=16)
# plt.legend()
# plt.savefig('beta_{}_sim.pdf'.format(neighbors))
# plt.show()

# epsilon = 1e-8
# beta = 0.0002
#
# print('EFD beta upper bound range: ', "({}, {})".format(min(EFD_upper), max(EFD_upper)), '\n')
# print('DCD beta upper bound: ', DCD_upper[0], '\n')
#
# dd = ((1-rho)**2) - 4*(mu**2)*beta
# dn = beta * (8*(mu**2)+4*((1-rho)**2))
# DCD_version = dn / dd
#
# # new_dcs = [0.5]
# print('new dc range: ', new_dcs, '\n')
# RED_upper_bound = []
# dc_plot = []
# for i in range(len(new_dcs)):
#     dc = new_dcs[i]
#     # beta = EFD_upper[i]
#     denominator = (1-rho)**2 - (beta * (mu*(np.sqrt(2)*dc+1)+np.sqrt(2)*dc*(1-rho))**2)
#     numerator = beta * (mu+(1-rho)) * (mu*(np.sqrt(2)*dc+1)+np.sqrt(2)*dc*(1-rho))
#     coefficient = (1+dc**2) / dc
#     if denominator > 0:  # The requirement for valued upper bound
#         dc_plot.append(dc)
#         RED_upper_bound.append(coefficient*numerator / denominator)
#         # print(dc, coefficient*numerator / denominator, '\n')
# DCD_upper_bound = [DCD_version for i in range(len(dc_plot))]
# min_upper_bound = RED_upper_bound[RED_upper_bound.index(min(RED_upper_bound))]
# dc_min_upper_bound = dc_plot[RED_upper_bound.index(min(RED_upper_bound))]
#
# print('DCD upper bound: ', DCD_upper_bound[0])
# print('minimum DED upper bound: ', min_upper_bound)
# print('Boosted: ', round(DCD_upper_bound[0] / min_upper_bound, 3))
# plt.scatter(dc_min_upper_bound, min_upper_bound)
# plt.plot(dc_plot, RED_upper_bound, label='DEED')
# plt.plot(dc_plot, DCD_upper_bound, label='DCD')
# plt.xlabel(r'value of $\gamma$', fontsize=16)
# plt.ylabel(r'$C_1$ value (related to upper bound)', fontsize=16)
# plt.legend()
# plt.savefig('upper_bound_sim.pdf')
# plt.show()

"""beta upper bound and upper bound comparison with given topology END"""

"""sub-optimal convergence upper bound in beta range START"""

# epsilon = 1e-8
# betas = np.arange(0, DCD_upper[0], 0.0005)[1:]
# print('Beta range: ', betas)
# print('New dc range: ', new_dcs)
#
# DCD_convergence_upper_bound = []
# RED_convergence_upper_bound = []
# for beta in betas:
#     dd = ((1-rho)**2) - 4*(mu**2)*beta
#     dn = beta * (8*(mu**2)+4*((1-rho)**2))
#     DCD_version = dn / dd
#     DCD_convergence_upper_bound.append(DCD_version)
#
#     RED_upper_bound = []
#     dc_plot = []
#     for i in range(len(new_dcs)):
#         dc = new_dcs[i]
#         # beta = EFD_upper[i]
#         denominator = (1-rho)**2 - (beta * (mu*(np.sqrt(2)*dc+1)+np.sqrt(2)*dc*(1-rho))**2)
#         numerator = beta * (mu+(1-rho)) * (mu*(np.sqrt(2)*dc+1)+np.sqrt(2)*dc*(1-rho))
#         coefficient = (1+dc**2) / dc
#         if denominator > 0:  # The requirement for valued upper bound
#             dc_plot.append(dc)
#             RED_upper_bound.append(coefficient*numerator / denominator)
#     RED_convergence_upper_bound.append(min(RED_upper_bound))
#
# plt.plot(betas, DCD_convergence_upper_bound, label='DCD')
# plt.plot(betas, RED_convergence_upper_bound, label='DEED')
# plt.ylim([0, max(RED_convergence_upper_bound)+1])
# plt.xlabel(r'value of $\beta$', fontsize=16)
# plt.ylabel(r'$C_1$ value (related to upper bound)', fontsize=16)
# plt.legend()
# plt.savefig('C_vs_beta.pdf')
# plt.show()

"""sub-optimal convergence upper bound in beta range END"""

# alpha = 0.5
# classes = 10
# clients = 20
# size = 6000
# Alpha = [alpha for i in range(classes)]
# # Generate samples from the Dirichlet distribution
# samples = np.random.dirichlet(Alpha, size=clients)
# # Print the generated samples
# num_samples = []
# for sample in samples:
#     sample = np.array(sample) * size
#     num_samples.append([int(round(i, 0)) for i in sample])
# print(num_samples)

import networkx as nx
import matplotlib.pyplot as plt

# Create a graph object
G = nx.Graph()

# Add nodes
G.add_node("node 1")
G.add_node("node 2")
G.add_node("node 3")
G.add_node("node 4")
G.add_node("node 5")
G.add_node("node 6")

# Add edges
G.add_edge("node 1", "node 2")
G.add_edge("node 1", "node 6")
G.add_edge("node 2", "node 1")
G.add_edge("node 2", "node 3")
G.add_edge("node 3", "node 2")
G.add_edge("node 3", "node 4")
G.add_edge("node 4", "node 3")
G.add_edge("node 4", "node 5")
G.add_edge("node 5", "node 4")
G.add_edge("node 5", "node 6")
G.add_edge("node 6", "node 5")
G.add_edge("node 6", "node 1")

# Draw the graph
nx.draw(G, with_labels=True)
plt.show()

