import copy
import numpy as np
import torch
import random

class Algorithms:
    def __init__(self, name=None, iter_round=None, device=None,
                 data_transform=None, num_clients=None, client_weights=None,
                 client_residuals=None, client_accumulates=None, client_compressors=None,
                 models=None, data_loaders=None, transfer=None, neighbor_models=None,
                 neighbors_accumulates=None, client_tmps=None, neighbors_estimates=None,
                 client_partition=None, control=False, alpha_max=None, compression_method=None):
        super().__init__()
        self.algorithm_name = name
        self.local_iter = iter_round
        self.device = device
        self.data_transform = data_transform
        self.num_clients = num_clients

        self.client_weights = client_weights
        self.client_residuals = client_residuals
        self.client_accumulates = client_accumulates
        self.client_compressor = client_compressors
        self.models = models
        self.data_loaders = data_loaders
        self.transfer = transfer
        self.neighbors = self.transfer.neighbors
        self.neighbor_models = neighbor_models
        self.neighbor_accumulates = neighbors_accumulates
        self.client_tmps = client_tmps
        self.neighbors_estimates = neighbors_estimates
        self.client_partition = client_partition
        self.client_history = self.client_residuals

        self.control = control

        # Testing parameter
        self.Alpha = []
        self.alpha_max = alpha_max
        self.compression_method = compression_method
        self.changes_ratio = []

        self.logger()

    def logger(self):
        print(' compression method:', self.compression_method, '\n',
              'apply control method: ', self.control, '\n',
              'alpha upper bound in theory: ', self.alpha_max, '\n',
              'running algorithm: ', self.algorithm_name, '\n')

    def _training(self, data_loader, client_weights, model):
        model.assign_weights(weights=client_weights)
        model.model.train()
        for i in range(self.local_iter):
            images, labels = next(iter(data_loader))
            images, labels = images.to(self.device), labels.to(self.device)
            if self.data_transform is not None:
                images = self.data_transform(images)

            model.optimizer.zero_grad()
            pred = model.model(images)
            # print(pred, len(pred))
            loss = model.loss_function(pred, labels)
            loss.backward()
            model.optimizer.step()
        trained_model = model.get_weights()
        return trained_model

    def _average_updates_EFD(self, updates, update):
        Averaged_weights = []
        for i in range(self.num_clients):
            Averaged_weights.append(sum(updates[i]) / len(updates[i]) - update[i])
        return Averaged_weights

    def _average_updates(self, updates):
        Averaged_weights = []
        for i in range(self.num_clients):
            Averaged_weights.append(sum(updates[i]) / len(updates[i]))
        return Averaged_weights

    def _check_weights(self, client_weights, neighbors_weights):
        checks = 0
        for n in range(self.num_clients):
            neighbors = self.neighbors[n]
            neighbors_models = neighbors_weights[n]

            check = 0
            for m in range(len(neighbors)):
                if torch.equal(neighbors_models[m], client_weights[neighbors[m]]):
                    check += 1
                else:
                    pass
            if check == len(self.neighbors[n]):
                checks += 1
            else:
                pass
        if checks == self.num_clients:
            return True
        else:
            return False

    def EFD(self, iter_num):
        Averaged_weights = self._average_updates_EFD(updates=self.neighbor_models, update=self.client_weights)  # X_t(W-I)
        alpha = []
        for n in range(self.num_clients):
            if self.control:
                qt = self.client_partition[n].get_q(iter_num)
                if np.random.binomial(1, qt) == 1:
                    Vector_update = self._training(data_loader=self.data_loaders[n], client_weights=self.client_weights[n], model=self.models[n])
                    Vector_update -= self.client_weights[n]  # gradient
                    Vector_update += Averaged_weights[n]
                else:
                    # pass
                    Vector_update = Averaged_weights[n]
            else:
                Vector_update = self._training(data_loader=self.data_loaders[n], client_weights=self.client_weights[n], model=self.models[n])
                Vector_update -= self.client_weights[n]  # gradient
                Vector_update += Averaged_weights[n]

            # Vector_update -= self.client_weights[n]  # Difference between averaged weights and local weights
            Vector_update -= self.client_history[n]

            # current_gradient = Vector_update
            # current_norm = torch.sum(torch.square(Vector_update)).item()
            # past_norm = torch.sum(torch.square(Vector_update + self.client_residuals[n])).item()
            # if past_norm == 0:
            #     current_vs_past = 0
            # else:
            #     current_vs_past = current_norm / past_norm
            # change_ratio.append(current_vs_past)

            bt_norm = torch.sum(torch.square(Vector_update + self.client_residuals[n])).item()
            Vector_update, self.client_residuals[n] = self.client_compressor[n].get_trans_bits_and_residual(iter=iter_num, w_tmp=Vector_update,
                                                                                                            w_residual=self.client_residuals[n],
                                                                                                            device=self.device, neighbors=self.neighbors[n])

            residual_norm = torch.sum(torch.square(self.client_residuals[n])).item()
            current_alpha = residual_norm / bt_norm
            alpha.append(current_alpha)

            self.client_weights[n] += Vector_update
            self.client_history[n] = Averaged_weights[n]
            for m in range(self.num_clients):
                if n in self.neighbors[m]:
                    self.neighbor_models[m][self.neighbors[m].index(n)] += Vector_update

        # if self._check_weights(self.client_weights, self.neighbor_models):
        #     print(iter_num, 'ALGORITHM SUCCESS THIS ROUND')
        # else:
        #     raise Exception('ALGORITHM FAIL')

        # print(alpha)
        if len(alpha) > 0:
            self.Alpha.append(sum(alpha)/len(alpha))

    def EFD_dc(self, iter_num):
        # Averaged_weights = self._average_updates_EFD(updates=self.neighbor_models, update=self.client_weights)  # X_t(W-I)
        Averaged_weights = self._average_updates(updates=self.neighbor_models)
        alpha = []
        for n in range(self.num_clients):
            if self.control:
                qt = self.client_partition[n].get_q(iter_num)
                if np.random.binomial(1, qt) == 1:
                    Vector_update = self._training(data_loader=self.data_loaders[n],
                                                   client_weights=self.client_weights[n], model=self.models[n])
                    Vector_update -= self.client_weights[n]  # gradient
                    Vector_update += Averaged_weights[n]
                else:
                    Vector_update = Averaged_weights[n]
            else:
                Vector_update = self._training(data_loader=self.data_loaders[n],
                                               client_weights=self.client_weights[n], model=self.models[n])
                Vector_update -= self.client_weights[n]  # gradient
                Vector_update += Averaged_weights[n]

            Vector_update -= self.client_weights[n]  # Difference between averaged weights and local weights

            bt_norm = torch.sum(torch.square(Vector_update + self.client_residuals[n])).item()

            Vector_update, self.client_residuals[n] = self.client_compressor[n].get_trans_bits_and_residual(iter=iter_num, w_tmp=Vector_update, w_residual=self.client_residuals[n], device=self.device, neighbors=self.neighbors[n])

            residual_norm = torch.sum(torch.square(self.client_residuals[n])).item()
            current_alpha = residual_norm / bt_norm
            alpha.append(current_alpha)

            self.client_weights[n] += Vector_update
            for m in range(self.num_clients):
                if n in self.neighbors[m]:
                    self.neighbor_models[m][self.neighbors[m].index(n)] += Vector_update

        # if self._check_weights(self.client_weights, self.neighbor_models):
        #     print(iter_num, 'ALGORITHM SUCCESS THIS ROUND')
        # else:
        #     raise Exception('ALGORITHM FAIL')

        # print(alpha)
        if len(alpha) > 0:
            self.Alpha.append(sum(alpha) / len(alpha))

    def _averaged_choco(self, updates, update):
        Averaged = []
        for i in range(self.num_clients):
            summation = torch.zeros_like(update[0])
            for j in range(len(updates[i])):
                summation += (1/len(updates[i])) * (updates[i][j] - update[i])
            Averaged.append(summation)
        return Averaged

    def CHOCO(self, iter_num, consensus):

        Averaged_accumulate = self._averaged_choco(updates=self.neighbor_accumulates, update=self.client_accumulates)

        for n in range(self.num_clients):
            if self.control:
                qt = self.client_partition[n].get_q(iter_num)
                if np.random.binomial(1, qt) == 1:
                    self.client_weights[n] = self.client_tmps[n] + consensus * Averaged_accumulate[n]
                    self.client_tmps[n] = self._training(data_loader=self.data_loaders[n], client_weights=self.client_weights[n], model=self.models[n])
                else:
                    self.client_weights[n] = self.client_tmps[n]
                    self.client_tmps[n] = self.client_weights[n]
            else:
                self.client_weights[n] = self.client_tmps[n] + consensus * Averaged_accumulate[n]
                self.client_tmps[n] = self._training(data_loader=self.data_loaders[n], client_weights=self.client_weights[n], model=self.models[n])

            Vector_update = self.client_weights[n] - self.client_accumulates[n]
            Vector_update, _ = self.client_compressor[n].get_trans_bits_and_residual(w_tmp=Vector_update, iter=iter_num,
                                                                                     w_residual=self.client_residuals[n],
                                                                                     device=self.device, neighbors=self.neighbors[n])
            self.client_accumulates[n] += Vector_update  # Vector Update is q_t
            for m in range(self.num_clients):
                if n in self.neighbors[m]:
                    self.neighbor_accumulates[m][self.neighbors[m].index(n)] += Vector_update

        if iter_num % 200 == 0:
            if consensus > 0.1:
                consensus -= 0.1
            else:
                consensus = consensus

    # def CHOCO_E(self, iter_num, consensus):  # Need to be revised
    #
    #     Averaged_accumulate = self._averaged_choco(updates=self.neighbor_accumulates, update=self.client_accumulates)
    #
    #     for n in range(self.num_clients):
    #         if iter_num >= 1:
    #             self.client_history[n] = self.client_weights[n]
    #         if self.control:
    #             print(self.control)
    #             qt = self.client_partition[n].get_q(iter_num)
    #             if np.random.binomial(1, qt) == 1:
    #                 self.client_weights[n] = self.client_tmps[n] + consensus * Averaged_accumulate[n]
    #                 self.client_tmps[n] = self._training(data_loader=self.data_loaders[n],
    #                                                      client_weights=self.client_weights[n], model=self.models[n])
    #             else:
    #                 self.client_weights[n] = self.client_tmps[n]
    #                 self.client_tmps[n] = self.client_weights[n]
    #         else:
    #             self.client_weights[n] = self.client_tmps[n] + consensus * Averaged_accumulate[n]
    #             self.client_tmps[n] = self._training(data_loader=self.data_loaders[n],
    #                                                  client_weights=self.client_weights[n], model=self.models[n])
    #
    #         Vector_update = self.client_weights[n] - self.client_history[n]
    #         Vector_update, self.client_residuals[n] = self.client_compressor[n].get_trans_bits_and_residual(w_tmp=Vector_update, iter=iter_num,
    #                                                                                                         w_residual=self.client_residuals[n],
    #                                                                                                         device=self.device, neighbors=self.neighbors[n])
    #         self.client_accumulates[n] += Vector_update  # Vector Update is q_t
    #         for m in range(self.num_clients):
    #             if n in self.neighbors[m]:
    #                 self.neighbor_accumulates[m][self.neighbors[m].index(n)] += Vector_update
    #
    #         if iter_num % 200 == 0:
    #             if consensus > 0.1:
    #                 consensus -= 0.1
    #             else:
    #                 consensus = consensus

    def DCD(self, iter_num):
        Averaged_weights = self._average_updates(updates=self.neighbor_models)

        for n in range(self.num_clients):
            if self.control:
                qt = self.client_partition[n].get_q(iter_num)
                if np.random.binomial(1, qt) == 1:
                    Vector_update = self._training(data_loader=self.data_loaders[n],
                                                   client_weights=self.client_weights[n],
                                                   model=self.models[n])
                    Vector_update -= self.client_weights[n]  # gradient
                    Vector_update += Averaged_weights[n]
                else:
                    Vector_update = Averaged_weights[n]
            else:
                Vector_update = self._training(data_loader=self.data_loaders[n],
                                               client_weights=self.client_weights[n],
                                               model=self.models[n])
                Vector_update -= self.client_weights[n]  # gradient
                Vector_update += Averaged_weights[n]

            Vector_update -= self.client_weights[n]  # Difference between averaged weights and local weights

            Vector_update, _ = self.client_compressor[n].get_trans_bits_and_residual(iter=iter_num,
                                                                                     w_tmp=Vector_update,
                                                                                     w_residual=
                                                                                     self.client_residuals[n],
                                                                                     device=self.device,
                                                                                     neighbors=self.neighbors[n])
            self.client_weights[n] += Vector_update
            for m in range(self.num_clients):
                if n in self.neighbors[m]:
                    self.neighbor_models[m][self.neighbors[m].index(n)] += Vector_update

    def ECD(self, iter_num):
        Averaged_weights = self._average_updates(updates=self.neighbors_estimates)

        for n in range(self.num_clients):
            current_weights = self.client_weights[n]
            if self.control:
                qt = self.client_partition[n].get_q(iter_num)
                if np.random.binomial(1, qt) == 1:
                    Vector_update = self._training(data_loader=self.data_loaders[n],
                                                   client_weights=self.client_weights[n],
                                                   model=self.models[n])
                    Vector_update -= self.client_weights[n]  # gradient
                    Vector_update += Averaged_weights[n]
                else:
                    Vector_update = Averaged_weights[n]
            else:
                Vector_update = self._training(data_loader=self.data_loaders[n],
                                               client_weights=self.client_weights[n],
                                               model=self.models[n])
                Vector_update -= self.client_weights[n]  # gradient
                Vector_update += Averaged_weights[n]

            self.client_weights[n] = Vector_update

            z_vector = (1 - 0.5 * iter_num) * current_weights + 0.5 * iter_num * Vector_update
            z_vector, _ = self.client_compressor[n].get_trans_bits_and_residual(iter=iter_num, w_tmp=z_vector,
                                                                                w_residual=self.client_residuals[n],
                                                                                device=self.device, neighbors=self.neighbors[n])
            for m in range(self.num_clients):
                if n in self.neighbors[m]:
                    self.neighbors_estimates[m][self.neighbors[m].index(n)] = (1 - (2/iter_num)) * self.neighbors_estimates[m][self.neighbors[m].index(n)] + (2/iter_num) * z_vector

