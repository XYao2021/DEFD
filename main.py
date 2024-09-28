import matplotlib.pyplot as plt
import torch
import random
import copy
import numpy as np
from torch.utils.data import DataLoader
from model.model import Model
from util.util import *
from compression import *
from partition import *
from config import *
from dataset.dataset import *
from trans_matrix import *
import time
from datetime import date
import os
from algorithms.algorithms import Algorithms


if device != 'cpu':
    current_device = torch.cuda.current_device()
    torch.cuda.set_device(current_device)

# device = 'cuda:{}'.format(CUDA_ID)

if __name__ == '__main__':
    ACC = []
    LOSS = []
    COMM = []
    ALPHAS = []
    MAXES = []

    for seed in Seed_set:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        train_data, test_data = loading(dataset_name=dataset, data_path=dataset_path, device=device)
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE_TEST, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_data, batch_size=BATCH_SIZE_TEST, shuffle=False, num_workers=0)

        Sample = Sampling(num_client=CLIENTS, num_class=len(train_data.classes), train_data=train_data, method='uniform', seed=seed)
        if DISTRIBUTION == 'Dirichlet':
            if ALPHA == 0:
                client_data = Sample.DL_sampling_single()
            elif ALPHA > 0:
                client_data = Sample.Synthesize_sampling(alpha=ALPHA)
        else:
            raise Exception('This data distribution method has not been embedded')

        client_train_loader = []
        client_residual = []
        client_compressor = []
        client_partition = []
        Models = []
        client_weights = []
        client_tmps = []
        client_accumulate = []
        neighbor_models = []
        neighbors_accumulates = []
        neighbors_estimates = []
        neighbor_updates = []
        estimate_gossip_error = []
        current_weights = []
        m_hat = []

        if ALGORITHM == 'EFD':
            max_value = 0.5642
            min_value = -0.5123
        elif ALGORITHM == 'CHOCO':
            max_value = 0.30123514
            min_value = -0.21583036
        elif ALGORITHM == 'AdaG':
            max_value = 0.30123514
            min_value = -0.21583036
        elif ALGORITHM == 'QSADDLe':
            max_value = 0.30123514
            min_value = -0.21583036

        # print(min_value, max_value)

        Transfer = Transform(num_nodes=CLIENTS, num_neighbors=NEIGHBORS, seed=seed, network=NETWORK)
        check = Check_Matrix(CLIENTS, Transfer.matrix)
        if check != 0:
            raise Exception('The Transfer Matrix Should be Symmetric')
        else:
            print('Transfer Matrix is Symmetric Matrix', '\n')
        alpha_max = Transfer.Get_alpha_upper_bound_theory()

        test_model = Model(random_seed=seed, learning_rate=LEARNING_RATE, model_name=model_name, device=device, flatten_weight=True, pretrained_model_file=load_model_file)
        # Preparation for every vector variables
        for n in range(CLIENTS):
            model = Model(random_seed=seed, learning_rate=LEARNING_RATE, model_name=model_name, device=device, flatten_weight=True, pretrained_model_file=load_model_file)
            Models.append(model)
            client_weights.append(model.get_weights())
            client_train_loader.append(DataLoader(client_data[n], batch_size=BATCH_SIZE, shuffle=True))
            client_residual.append(torch.zeros_like(model.get_weights()).to(device))
            neighbor_models.append([model.get_weights() for i in range(len(Transfer.neighbors[n]))])

            neighbor_updates.append([torch.zeros_like(model.get_weights()) for i in range(len(Transfer.neighbors[n]))])

            if COMPRESSION == 'quantization':
                if CONTROL is True:
                    client_compressor.append(Lyapunov_compression_Q(node=n, avg_comm_cost=average_comm_cost, V=V, W=W, max_value=max_value, min_value=min_value))
                    client_partition.append(Lyapunov_Participation(node=n, average_comp_cost=average_comp_cost, V=V, W=W, seed=seed))
                else:
                    if FIRST is True:
                        client_compressor.append(Quantization_I(num_bits=QUANTIZE_LEVEL, max_value=max_value, min_value=min_value, device=device))
                    else:
                        client_compressor.append(Quantization_U(num_bits=QUANTIZE_LEVEL, max_value=max_value, min_value=min_value, device=device, discount=DISCOUNT))

            elif COMPRESSION == 'topk':
                if CONTROL is True:
                    client_compressor.append(Lyapunov_compression_T(node=n, avg_comm_cost=average_comm_cost, V=V, W=W))
                    client_partition.append(Lyapunov_Participation(node=n, average_comp_cost=average_comp_cost, V=V, W=W, seed=seed))
                else:
                    client_compressor.append(Top_k(ratio=RATIO, device=device))
            else:
                raise Exception('Unknown compression method, please write the compression method first')

            if ALGORITHM == 'CHOCO':
                client_tmps.append(model.get_weights().to(device))
                client_accumulate.append(torch.zeros_like(model.get_weights()).to(device))
                neighbors_accumulates.append([torch.zeros_like(model.get_weights()).to(device) for i in range(len(Transfer.neighbors[n]))])
            if ALGORITHM == 'AdaG':
                client_tmps.append(model.get_weights().to(device))
                client_accumulate.append(torch.zeros_like(model.get_weights()).to(device))
                neighbors_accumulates.append([torch.zeros_like(model.get_weights()).to(device) for i in range(len(Transfer.neighbors[n]))])
                estimate_gossip_error.append(torch.zeros_like(model.get_weights()).to(device))
            if ALGORITHM == 'QSADDLe':
                current_weights.append(model.get_weights().to(device))
                client_tmps.append(model.get_weights().to(device))
                client_accumulate.append(torch.zeros_like(model.get_weights()).to(device))
                neighbors_accumulates.append([torch.zeros_like(model.get_weights()).to(device) for i in range(len(Transfer.neighbors[n]))])
                m_hat.append(torch.zeros_like(model.get_weights()).to(device))

        Algorithm = Algorithms(name=ALGORITHM, iter_round=ROUND_ITER, device=device, data_transform=data_transform,
                               num_clients=CLIENTS, client_weights=client_weights, client_residuals=client_residual,
                               client_accumulates=client_accumulate, client_compressors=client_compressor,
                               models=Models, data_loaders=client_train_loader, transfer=Transfer,
                               neighbor_models=neighbor_models, neighbors_accumulates=neighbors_accumulates,
                               client_tmps=client_tmps, neighbors_estimates=neighbors_estimates, client_partition=client_partition,
                               control=CONTROL, alpha_max=alpha_max, compression_method=COMPRESSION,
                               estimate_gossip_error=estimate_gossip_error, current_weights=current_weights, m_hat=m_hat)
        global_loss = []
        Test_acc = []
        iter_num = 0
        print(ALGORITHM)

        while True:
            # print('SEED ', '|', seed, '|', 'ITERATION ', iter_num)
            if ALGORITHM == 'EFD':  # main algorithm
                if DISCOUNT == 0.0:  # Equals to DCD
                    if iter_num == 0:
                        print('0', DISCOUNT)
                    Algorithm.DCD(iter_num=iter_num)
                else:
                    if iter_num == 0:
                        print('1', DISCOUNT)
                    Algorithm.EFD_dc(iter_num=iter_num)
            elif ALGORITHM == 'CHOCO':
                if iter_num == 0:
                    print('2', CONSENSUS_STEP)
                Algorithm.CHOCO(iter_num=iter_num, consensus=CONSENSUS_STEP)
            elif ALGORITHM == 'AdaG':  # beta = 0.999 consensus = 0.002 (0.1)
                if iter_num == 0:
                    print('3', CONSENSUS_STEP)
                Algorithm.AdaG_SGD(iter_num=iter_num+1, beta=0.999, consensus=CONSENSUS_STEP, epsilon=1e-8)
            elif ALGORITHM == 'QSADDLe':
                if iter_num == 0:
                    print('4', CONSENSUS_STEP)
                Algorithm.Comp_QSADDLe(iter_num=iter_num, rho=0.01, beta=0.9, learning_rate=LEARNING_RATE, consensus=CONSENSUS_STEP, mu=0.9)
            else:
                raise Exception('Unknown algorithm, please update the algorithm codes')

            iter_num += 1
            # for i in range(CLIENTS):
            #     print(iter_num, i, Algorithm.models[i].get_weights())
            test_weights = average_weights([Algorithm.models[i].get_weights() for i in range(CLIENTS)])
            # test_weights = average_weights(weights=Algorithm.client_weights)
            train_loss, train_acc = test_model.accuracy(weights=test_weights, test_loader=train_loader, device=device)
            test_loss, test_acc = test_model.accuracy(weights=test_weights, test_loader=test_loader, device=device)

            global_loss.append(train_loss)
            Test_acc.append(test_acc)
            print('SEED |', seed, '| iteration |', iter_num, '| Global Loss', train_loss, '| Training Accuracy |',
                  train_acc, '| Test Accuracy |', test_acc, '\n')

            if iter_num >= AGGREGATION:
                ACC += Test_acc
                LOSS += global_loss
                break
        del Models
        del client_weights

        torch.cuda.empty_cache()  # Clean the memory cache

    if STORE == 1:
        if FIRST is True:
            Maxes = []
            Mines = []
            for i in range(CLIENTS):
                Maxes.append(max(client_compressor[i].max))
                Mines.append(min(client_compressor[i].min))
            txt_list = [Maxes, '\n', Mines, '\n', ACC, '\n', LOSS]
        else:
            txt_list = [ACC, '\n', LOSS]

        if COMPRESSION == 'quantization':
            f = open('{}|{}|{}|{}|{}|{}|{}|{}|.txt'.format(ALGORITHM, ALPHA, QUANTIZE_LEVEL, DISCOUNT, CONTROL, dataset, LEARNING_RATE, CONSENSUS_STEP, date.today(), time.strftime("%H:%M:%S", time.localtime())), 'w')
        elif COMPRESSION == 'topk':
            f = open('{}|{}|{}|{}|{}|{}|{}|{}|.txt'.format(ALGORITHM, ALPHA, RATIO, DISCOUNT, CONTROL, dataset, LEARNING_RATE, CONSENSUS_STEP, date.today(), time.strftime("%H:%M:%S", time.localtime())), 'w')
        else:
            raise Exception('Unknown compression method')

        for item in txt_list:
            f.write("%s\n" % item)
    else:
        print('NOT STORE THE RESULTS THIS TIME')
