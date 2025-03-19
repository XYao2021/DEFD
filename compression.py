import copy
import random
from abc import ABC
import numpy.random
import torch
import numpy as np
import abc
from config import *
# from numpy.random import RandomState, SeedSequence
# from numpy.random import MT19937


# def communication_cost(node, iter, full_size, trans_size):
#     if trans_size == 0:
#         return 0.0
#     else:
#         rs = RandomState(MT19937(SeedSequence(iter * 800 + node + 23456)))
#         rs1 = RandomState(MT19937(SeedSequence(iter * 800 + node + 23457)))
#
#         constant = 0.05  # beta
#         SNR_0 = rs.chisquare(df=2)
#         SNR_1 = rs1.chisquare(df=2)
#
#         if SNR_0 > SNR_1:
#             gamma = 1 / np.log2(1 + SNR_1)
#         elif SNR_1 > SNR_0:
#             gamma = 1 / np.log2(1 + SNR_0)
#         return constant + float(trans_size)/full_size * gamma
#
# def communication_cost_quan(node, iter, full_size, trans_size):
#     if trans_size == 0:
#         return 0.0
#     else:
#         rs = RandomState(MT19937(SeedSequence(iter * 800 + node + 23456)))
#         rs1 = RandomState(MT19937(SeedSequence(iter * 800 + node + 23457)))
#
#         constant = 16  # beta
#         SNR_0 = rs.chisquare(df=2)
#         SNR_1 = rs1.chisquare(df=2)
#
#         if SNR_0 > SNR_1:
#             gamma = 1 / np.log2(1 + SNR_1)
#         elif SNR_1 > SNR_0:
#             gamma = 1 / np.log2(1 + SNR_0)
#
#         return constant + trans_size * gamma
def get_snr(iter, node, neighbors):
    SNRs = []
    for i in range(len(neighbors)):
        if neighbors[i] == node:
            rs = RandomState(MT19937(SeedSequence(iter * 800 + node + 23456)))
        else:
            rs = RandomState(MT19937(SeedSequence(iter * 800 + node + 23456 + i)))
        SNRs.append(rs.chisquare(df=2))
    return min(SNRs)

def communication_cost(node, iter, full_size, trans_size):
    if trans_size == 0:
        return 0.0
    else:
        rs = RandomState(MT19937(SeedSequence(iter * 800 + node + 23456)))

        constant = 0.05  # beta
        SNR_0 = rs.chisquare(df=2)

        gamma = 1 / np.log2(1 + SNR_0)
        return constant + (float(trans_size)/full_size) * gamma

def communication_cost_quan(node, iter, full_size, trans_size):
    if trans_size == 0:
        return 0.0
    else:
        rs = RandomState(MT19937(SeedSequence(iter * 800 + node + 23456)))

        constant = 8 / full_size  # beta
        SNR_0 = rs.chisquare(df=2)

        gamma = 1 / np.log2(1 + SNR_0)

        return constant + (trans_size/full_size) * gamma

def communication_cost_multiple(node, iter, full_size, trans_size, channel_quality):
    if trans_size == 0:
        return 0.0
    else:
        constant = 0.05  # beta
        # print('neighbors', channel_quality)
        if channel_quality is None:
            rs = RandomState(MT19937(SeedSequence(iter * 800 + node + 23456)))
            SNR_0 = rs.chisquare(df=2)
            gamma = 1 / np.log2(1 + SNR_0)
        else:
            neighbors, add_trans = channel_quality
            neighbors = np.setdiff1d(neighbors, node)
            SNR = []
            Gamma_neighbor = []
            for i in range(len(neighbors)):
                rs = RandomState(MT19937(SeedSequence(iter * 800 + node + 23456 + neighbors[i])))
                SNR.append(rs.chisquare(df=2))
                gamma_neighbor = []
                for k in range(len(add_trans[i])):
                    rs1 = RandomState(MT19937(SeedSequence(iter * 800 + neighbors[i] + 23456 + add_trans[i][k])))
                    gamma_neighbor.append(1 / np.log2(1 + rs1.chisquare(df=2)))
                Gamma_neighbor.append(sum(gamma_neighbor))
            SNR = min(SNR)
            gamma = 1 / np.log2(1 + SNR)
            gamma += sum(Gamma_neighbor)
            # for j in range(len(SNR_neighbors)):
            #     gamma += (1 * add_trans[j]) / np.log2(1 + SNR_neighbors[j])
        return constant * 2 + (float(trans_size) / full_size) * gamma

def communication_cost_quan_multiple(node, iter, full_size, trans_size, channel_quality):
    if trans_size == 0:
        return 0.0
    else:
        constant = 8 / full_size  # beta
        # print('neighbors', channel_quality)
        if channel_quality is None:
            rs = RandomState(MT19937(SeedSequence(iter * 800 + node + 23456)))
            SNR_0 = rs.chisquare(df=2)
            gamma = 1 / np.log2(1 + SNR_0)
        else:
            neighbors, add_trans = channel_quality
            neighbors = np.setdiff1d(neighbors, node)
            SNR = []
            Gamma_neighbor = []
            for i in range(len(neighbors)):
                rs = RandomState(MT19937(SeedSequence(iter * 800 + node + 23456 + neighbors[i])))
                SNR.append(rs.chisquare(df=2))
                gamma_neighbor = []
                for k in range(len(add_trans[i])):
                    rs1 = RandomState(MT19937(SeedSequence(iter * 800 + neighbors[i] + 23456 + add_trans[i][k])))
                    gamma_neighbor.append(1 / np.log2(1 + rs1.chisquare(df=2)))
                Gamma_neighbor.append(sum(gamma_neighbor))
            SNR = min(SNR)
            gamma = 1 / np.log2(1 + SNR)
            gamma += sum(Gamma_neighbor)
            # for j in range(len(SNR_neighbors)):
            #     gamma += (1 * add_trans[j]) / np.log2(1 + SNR_neighbors[j])
        return constant*2 + (float(trans_size) / full_size) * gamma

def broadcast_quan(node, iter, full_size, trans_size, snr):
    if trans_size == 0:
        return 0.0
    else:
        constant = 8 / full_size
        gamma = 1 / np.log2(1 + snr)
        return constant + (trans_size/full_size) * gamma

def broadcast_topk(node, iter, full_size, trans_size, snr):
    if trans_size == 0:
        return 0.0
    else:
        constant = 0.05
        gamma = 1 / np.log2(1 + snr)
        return constant + (trans_size/full_size) * gamma

class Compression(abc.ABC):
    def __init__(self, node):
        self.node = node

    def get_trans_bits_and_residual(self, iter, w_tmp, w_residual, device, channel_quality):  # w_tmp is gradient this time
        if w_tmp is None:
            w_tmp = w_residual  # w_residual is e_t
        else:
            w_tmp += w_residual

        trans_indices, not_trans_indices, trans_bits = self._get_trans_indices(iter, w_tmp, channel_quality)

        w_tmp_residual = copy.deepcopy(w_tmp)
        w_tmp[not_trans_indices] = 0  # transfer vector v_t, sparse vector
        w_tmp_residual -= w_tmp  # accumulate the residual for not transmit bits

        return w_tmp, w_tmp_residual

    def _get_trans_indices(self, iter, w_tmp, channel_quality):
        raise NotImplementedError()  #TODO: What does this mean?

class Compression_1(abc.ABC):
    def __init__(self, node):
        self.node = node

    def get_trans_bits_and_residual(self, iter, w_tmp, w_residual, device, neighbors):  # w_tmp is gradient this time
        discount_parameter = DISCOUNT
        if w_tmp is None:
            w_tmp = discount_parameter * w_residual  # w_residual is e_t
        else:
            w_tmp += discount_parameter * w_residual

        trans_indices, not_trans_indices = self._get_trans_indices(iter, w_tmp, neighbors)

        w_tmp_residual = copy.deepcopy(w_tmp)
        w_tmp[not_trans_indices] = 0  # transfer vector v_t, sparse vector
        w_tmp_residual -= w_tmp  # accumulate the residual for not transmit bits

        return w_tmp, w_tmp_residual

    def _get_trans_indices(self, iter, w_tmp, neighbors):
        raise NotImplementedError()  #TODO: What does this mean?

class Compression_Q(abc.ABC):
    def __init__(self, node):
        self.node = node

    def get_trans_bits_and_residual(self, iter, w_tmp, device, w_residual, neighbors):  # w_tmp is gradient this time
        if w_tmp is None:
            w_tmp = w_residual  # w_residual is e_t
        else:
            w_tmp += w_residual

        quantize_value, residual_value = self._get_trans_indices(iter, w_tmp, neighbors)

        return quantize_value, residual_value

    def _get_trans_indices(self, iter, w_tmp, neighbors):
        raise NotImplementedError()  #TODO: What does this mean?

# "Chose Different Compression Method"
class Lyapunov_compression(Compression):
    def __init__(self, node, avg_comm_cost, V, W):
        super().__init__(node)
        self.avg_comm_cost = avg_comm_cost
        self.V = V
        self.queue = W  # Initial queue length

    def _get_trans_indices(self, iter, w_tmp, channel_quality):
        # full_size = w_tmp.size()[0]
        full_size = w_tmp.shape[0]
        bt_square = torch.square(w_tmp)
        Bt = torch.sum(bt_square)
        bt_sq_sort, bt_sq_sort_indices = torch.sort(bt_square, descending=True)

        no_transmit_penalty = self.V * torch.sum(bt_square) - self.queue * self.avg_comm_cost
        cost_delta = self.queue * (communication_cost(self.node, iter, full_size, 2) - communication_cost(self.node, iter, full_size, 1))  # equal to gamma_t * PHI_t(queue at time t)

        tmp = torch.arange(bt_square.shape[0], device=device)
        tmp2 = tmp[self.V * bt_sq_sort <= cost_delta]
        if len(tmp2) > 0:
            j = tmp2[0]
            # print(self.node, len(tmp2), j)
        else:
            j = full_size

        drift_plus_penalty = self.V * torch.sum(bt_sq_sort[j:]) + self.queue * (communication_cost(self.node, iter, full_size, j) - self.avg_comm_cost)

        if drift_plus_penalty < no_transmit_penalty:
            trans_bits = j
        else:
            trans_bits = 0
        self.queue += communication_cost(self.node, iter, full_size, trans_bits) - self.avg_comm_cost
        self.queue = max(0.001, self.queue)  # Not allow to have the negative queues, set to very small one
        return bt_sq_sort_indices[:trans_bits], bt_sq_sort_indices[trans_bits:], trans_bits

class Lyapunov_compression_T(Compression_1):
    def __init__(self, node, avg_comm_cost, V, W):
        super().__init__(node)
        self.avg_comm_cost = avg_comm_cost
        self.V = V
        self.queue = W  # Initial queue length

    def _get_trans_indices(self, iter, w_tmp, neighbors):
        # full_size = w_tmp.size()[0]
        full_size = w_tmp.shape[0]
        bt_square = torch.square(w_tmp)
        Bt = torch.sum(bt_square)
        bt_sq_sort, bt_sq_sort_indices = torch.sort(bt_square, descending=True)
        # print(self.node, channel_quality)
        snr = get_snr(iter=iter, node=self.node, neighbors=neighbors)
        # print(iter, self.node, snr)

        no_transmit_penalty = self.V * torch.sum(bt_square) - self.queue * self.avg_comm_cost
        cost_delta = self.queue * (broadcast_topk(self.node, iter, full_size, 2, snr) - broadcast_topk(self.node, iter, full_size, 1, snr))  # equal to gamma_t * PHI_t(queue at time t)

        tmp = torch.arange(bt_square.shape[0], device=device)
        tmp2 = tmp[self.V * bt_sq_sort <= cost_delta]
        if len(tmp2) > 0:
            j = tmp2[0]
        else:
            j = full_size

        drift_plus_penalty = self.V * torch.sum(bt_sq_sort[j:]) + self.queue * (broadcast_topk(self.node, iter, full_size, j, snr) - self.avg_comm_cost)

        if drift_plus_penalty < no_transmit_penalty:
            trans_bits = j
        else:
            trans_bits = 0
        self.queue += broadcast_topk(self.node, iter, full_size, trans_bits, snr) - self.avg_comm_cost
        self.queue = max(0.001, self.queue)  # Not allow to have the negative queues, set to very small one
        return bt_sq_sort_indices[:trans_bits], bt_sq_sort_indices[trans_bits:]

class Lyapunov_compression_Q(Compression_Q):
    def __init__(self, node, avg_comm_cost, V, W, max_value, min_value):
        super().__init__(node)
        self.avg_comm_cost = avg_comm_cost
        self.V = V
        self.queue = W  # Initial queue length
        self.max_value = max_value
        self.min_value = min_value

    def _get_trans_indices(self, iter, w_tmp, neighbors):

        full_size = 12
        no_trans_cost = self.V * torch.sum(torch.square(w_tmp)) - self.queue * self.avg_comm_cost
        residual_value = w_tmp

        snr = get_snr(iter=iter, node=self.node, neighbors=neighbors)
        for trans_bits in [4, 6, 8, 10, 12]:  # [4, 6, 8, 10]
            scale = 2 ** trans_bits - 1
            step = (self.max_value - self.min_value) / scale

            centroids = []
            value = self.min_value
            centroids.append(value)
            while len(centroids) < 2 ** trans_bits:
                value = value + step
                centroids.append(value)

            centroids = torch.tensor(centroids).to(device)
            distances = torch.cdist(torch.reshape(w_tmp, (-1, 1)), torch.reshape(centroids, (-1, 1)))
            assignments = torch.argmin(distances, dim=1)

            quantize_value = torch.index_select(input=torch.tensor(centroids), dim=0, index=assignments)
            # trans_cost = self.V * torch.sum(torch.square((w_tmp - quantize_value))) + self.queue * (communication_cost_quan(self.node, iter, full_size, trans_bits) - self.avg_comm_cost)

            trans_cost = self.V * torch.sum(torch.square((w_tmp - quantize_value))) + self.queue * (broadcast_quan(self.node, iter, full_size, trans_bits, snr) - self.avg_comm_cost)

            if trans_cost < no_trans_cost:
                trans_bits = trans_bits
                quantize_value = quantize_value
                break
            else:
                trans_bits = 0
                quantize_value = torch.zeros_like(w_tmp)
        residual_value -= quantize_value

        self.queue += broadcast_quan(self.node, iter, full_size, trans_bits, snr) - self.avg_comm_cost
        self.queue = max(0.001, self.queue)  # Not allow to have the negative queues, set to very small one
        return quantize_value, residual_value


class Fixed_Compression(Compression):
    def __init__(self, node, avg_comm_cost, ratio=1.0):
        super().__init__(node)
        self.avg_comm_cost = avg_comm_cost
        self.ratio = ratio

    def _get_trans_indices(self, iter, w_tmp, channel_quality):
        full_size = w_tmp.size()[0]
        bt_square = torch.square(w_tmp)
        bt_square_sorted, bt_sorted_indices = torch.sort(bt_square, descending=True)

        no_trans_cost = communication_cost(self.node, iter, full_size, 0)
        if no_trans_cost > 0:
            raise Exception('No transmit cost should be zero')

        k = int(full_size * self.ratio)
        if k > torch.count_nonzero(bt_square).item():
            k = torch.count_nonzero(bt_square).item()
        trans_cost = communication_cost(self.node, iter, full_size, k)

        if trans_cost > 0:
            p_trans = min(1.0, self.avg_comm_cost / trans_cost)
        else:
            p_trans = 1.0

        if np.random.binomial(1, p_trans) == 1:
            trans_bits = k
        else:
            trans_bits = 0
        return bt_sorted_indices[:trans_bits], bt_sorted_indices[trans_bits:], trans_bits

class Top_k(abc.ABC):
    def __init__(self, ratio=1.0, device=None, discount=0.0):
        super().__init__()
        self.ratio = ratio
        self.device = device
        self.discount_parameter = discount

    def get_trans_bits_and_residual(self, iter, w_tmp, w_residual, device, neighbors):
        if w_tmp is None:
            w_tmp = self.discount_parameter * w_residual  # w_residual is e_t
        else:
            w_tmp += self.discount_parameter * w_residual

        full_size = w_tmp.size()[0]
        # print(iter, full_size, self.ratio)
        bt_square = torch.square(w_tmp)
        bt_square_sorted, bt_sorted_indices = torch.sort(bt_square, descending=True)
        trans_bits = int(self.ratio * full_size)

        trans_indices, not_trans_indices = bt_sorted_indices[:trans_bits], bt_sorted_indices[trans_bits:]
        w_tmp_residual = copy.deepcopy(w_tmp)
        w_tmp[not_trans_indices] = 0  # transfer vector v_t, sparse vector
        w_tmp_residual -= w_tmp
        return w_tmp, w_tmp_residual

class Quantization(abc.ABC):  # Biased quantization
    def __init__(self, num_bits=8, max_value=0, min_value=0, device=None, discount=0.0):
        self.device = device
        self.num_bits = num_bits
        self.scale = 2**self.num_bits - 1
        self.max_value = max_value
        self.min_value = min_value
        self.discount_parameter = discount
        if self.max_value == self.min_value == 0:
            raise Exception('Please set the max and min value for quantization')
        self._initialization()

    def _initialization(self):
        step = (self.max_value - self.min_value) / self.scale

        quantization = []
        value = self.min_value
        quantization.append(value)
        while len(quantization) < 2 ** self.num_bits:
            value = value + step
            quantization.append(value)
        self.quantization = torch.tensor(quantization).to(self.device)

    def get_trans_bits_and_residual(self, iter, w_tmp, w_residual, device, neighbors):
        if w_tmp is None:
            w_tmp = self.discount_parameter * w_residual  # w_residual is e_t
        else:
            w_tmp += self.discount_parameter * w_residual

        distances = torch.cdist(torch.reshape(w_tmp, (-1, 1)), torch.reshape(self.quantization, (-1, 1)))
        # a = torch.argsort(distances, dim=1)
        # print(iter, 'distance: ', distances, distances.size())
        assignments = torch.argmin(distances, dim=1)

        w_tmp_quantized = torch.index_select(input=torch.tensor(self.quantization), dim=0, index=assignments)
        w_residual = w_tmp - w_tmp_quantized
        return w_tmp_quantized, w_residual

class Quantization_U(abc.ABC):  # Unbiased quantization
    def __init__(self, num_bits=8, max_value=0, min_value=0, discount=0.0, device=None):
        self.device = device
        self.num_bits = num_bits
        self.scale = 2**self.num_bits - 1
        self.max_value = max_value
        self.min_value = min_value
        self.discount_parameter = discount
        if self.max_value == self.min_value == 0:
            raise Exception('Please set the max and min value for quantization')
        self._initialization()

    def _initialization(self):
        step = (self.max_value - self.min_value) / self.scale

        quantization = []
        value = self.min_value
        quantization.append(value)
        while len(quantization) < 2 ** self.num_bits:
            value = value + step
            quantization.append(value)
        self.quantization = torch.tensor(quantization).to(self.device)

    def get_trans_bits_and_residual(self, iter, w_tmp, w_residual, device, neighbors):
        # print(self.discount_factor)
        if w_tmp is None:
            w_tmp = self.discount_parameter * w_residual  # w_residual is e_t
        else:
            w_tmp += self.discount_parameter * w_residual

        distances = torch.cdist(torch.reshape(w_tmp, (-1, 1)), torch.reshape(self.quantization, (-1, 1)))

        sorted_distance_value = torch.sort(distances, dim=1).values
        sorted_distances_index = torch.argsort(distances, dim=1)

        first_choice_value = torch.flatten(sorted_distance_value[:, :1]).tolist()
        first_choice_index = torch.flatten(sorted_distances_index[:, :1])

        second_choice_index = torch.flatten(sorted_distances_index[:, 1:2])
        sorted_distance_value = sorted_distance_value[:, :2]

        summation = torch.sum(sorted_distance_value, dim=1).tolist()
        random_choice = np.random.uniform(high=np.array(summation))

        decision = random_choice > np.array(first_choice_value)

        assignments = copy.deepcopy(second_choice_index)
        indexes = torch.tensor(np.where(decision)[0])
        assignments[indexes] = first_choice_index[indexes]

        w_tmp_quantized = torch.index_select(input=torch.tensor(self.quantization), dim=0, index=assignments)
        w_residual = w_tmp - w_tmp_quantized
        return w_tmp_quantized, w_residual

class Quantization_I(abc.ABC):  # Initialize setup for max and min value
    def __init__(self, num_bits=4, max_value=0, min_value=0, device=None, discount=0.0):
        self.num_bits = num_bits
        self.scale = 2**self.num_bits - 1
        self.max_value = max_value
        self.min_value = min_value
        self.max = []
        self.min = []
        self.discount_parameter = discount

    def get_trans_bits_and_residual(self, iter, w_tmp, w_residual, device, neighbors):
        if w_tmp is None:
            w_tmp = self.discount_parameter * w_residual  # w_residual is e_t
        else:
            w_tmp += self.discount_parameter * w_residual
        max_value = torch.max(w_tmp)
        min_value = torch.min(w_tmp)
        self.max.append(max_value)
        self.min.append(min_value)
        # print(max_value, min_value)

        step = (max_value - min_value) / self.scale

        centroids = []
        value = min_value
        centroids.append(value)
        while len(centroids) < 2 ** self.num_bits:
            value = value + step
            centroids.append(value)

        centroids = torch.tensor(centroids).to(device)
        distances = torch.cdist(torch.reshape(w_tmp, (-1, 1)), torch.reshape(centroids, (-1, 1)))
        assignments = torch.argmin(distances, dim=1)

        w_tmp_quantized = torch.tensor([centroids[i] for i in assignments])
        w_residual = w_tmp - w_tmp_quantized
        return w_tmp_quantized, w_residual

class Quantization_U_1(abc.ABC):  # Unbiased quantization
    def __init__(self, num_bits=8, max_value=0, min_value=0, discount=0.0, device=None):
        self.device = device
        self.num_bits = num_bits
        self.scale = 2**self.num_bits - 1
        self.max_value = max_value
        self.min_value = min_value
        self.discount_parameter = discount
        if self.max_value == self.min_value == 0:
            raise Exception('Please set the max and min value for quantization')
        self._initialization()

    def _initialization(self):
        step = (self.max_value - self.min_value) / self.scale

        quantization = []
        value = self.min_value
        quantization.append(value)
        while len(quantization) < 2 ** self.num_bits:
            value = value + step
            quantization.append(value)
        self.quantization = torch.tensor(quantization).to(self.device)

    def get_trans_bits_and_residual(self, iter, w_tmp, w_residual, device, neighbors):
        # print(iter, self.discount_parameter)
        if w_tmp is None:
            w_tmp = self.discount_parameter * w_residual  # w_residual is e_t
        else:
            w_tmp += self.discount_parameter * w_residual

        distances = torch.cdist(torch.reshape(w_tmp, (-1, 1)), torch.reshape(self.quantization, (-1, 1)))

        sorted_distance_value = torch.sort(distances, dim=1).values
        sorted_distances_index = torch.argsort(distances, dim=1)

        first_choice_value = torch.flatten(sorted_distance_value[:, :1])
        first_choice_index = torch.flatten(sorted_distances_index[:, :1])

        second_choice_value = torch.flatten(sorted_distance_value[:, 1:2])
        second_choice_index = torch.flatten(sorted_distances_index[:, 1:2])

        # print(first_choice_value, second_choice_value)
        summation_1 = first_choice_value + second_choice_value

        choices_2 = second_choice_value / summation_1
        choices_2 = torch.bernoulli(choices_2)

        assignments = copy.deepcopy(first_choice_index)
        assignments[torch.where(choices_2 > 0)[0]] = second_choice_index[torch.where(choices_2 > 0)[0]]

        w_tmp_quantized = torch.index_select(input=torch.tensor(self.quantization), dim=0, index=assignments)
        w_residual = w_tmp - w_tmp_quantized
        return w_tmp_quantized, w_residual

class Rand_k(abc.ABC):
    def __init__(self, ratio=1.0, device=None, discount=0.0):
        super().__init__()
        self.ratio = ratio
        self.device = device
        self.discount_parameter = discount

    def get_trans_bits_and_residual(self, iter, w_tmp, w_residual, device, neighbors):
        if w_tmp is None:
            w_tmp = self.discount_parameter * w_residual  # w_residual is e_t
        else:
            w_tmp += self.discount_parameter * w_residual

        full_size = w_tmp.size()[0]
        # print(iter, full_size, self.ratio)
        trans_bits = int(self.ratio * full_size)
        indices = random.sample(range(full_size), trans_bits)

        w_trans = torch.zeros_like(w_tmp)
        w_tmp_residual = copy.deepcopy(w_tmp)

        w_trans[indices] = w_tmp[indices]
        w_tmp = w_trans  # transfer vector v_t, sparse vector
        w_tmp_residual -= w_tmp
        return w_tmp, w_tmp_residual
