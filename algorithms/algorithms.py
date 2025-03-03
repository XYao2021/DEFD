import copy
import numpy as np
import torch
import random
import copy
import time

def random_quantize_mat(X, s):
    # norm of vector: 32-bits
    # sign of vector entries: 1 * len(x) bits
    # s-bin interval: log(s+1) * len(x) bits
    # e.g. [0, .25, .5, .75, 1] as intervals, s=4
    # for x \in R^d: total = (32 + d * (1 + log(s+1))) bits
    d = X.shape[0]
    tau = 1 + min([d/s**2, np.sqrt(d)/s])
    signs = torch.sign(X)
    x_norms = torch.linalg.norm(X, ord=2, axis=0).reshape(-1, 1)
    # Q = np.abs(X)/x_norms * s + np.random.uniform(low=0, high=1, size=X.shape)
    Q = torch.abs(X) / x_norms * s
    Q = torch.floor(Q) * signs * x_norms / (s * tau)
    return Q

class Algorithms:
    def __init__(self, name=None, iter_round=None, device=None,
                 data_transform=None, num_clients=None, client_weights=None,
                 client_residuals=None, client_accumulates=None, client_compressors=None,
                 models=None, data_loaders=None, transfer=None, neighbor_models=None,
                 neighbors_accumulates=None, client_tmps=None, neighbors_estimates=None,
                 client_partition=None, control=False, alpha_max=None, compression_method=None,
                 estimate_gossip_error=None, current_weights=None, m_hat=None, adaptive=False, threshold=None,
                 H=None, G=None, neighbor_H=None, neighbor_G=None):
        super().__init__()
        self.algorithm_name = name
        self.local_iter = iter_round
        self.device = device
        self.data_transform = data_transform
        self.num_clients = num_clients

        self.client_weights = client_weights
        self.client_residuals = client_residuals
        self.client_compressor = client_compressors
        self.models = models
        self.data_loaders = data_loaders
        self.transfer = transfer
        self.neighbors = self.transfer.neighbors
        self.neighbor_models = neighbor_models
        self.client_partition = client_partition
        self.client_history = self.client_residuals
        self.initial_error_norm = None

        "CHOCO"
        self.client_accumulates = client_accumulates
        self.neighbor_accumulates = neighbors_accumulates
        self.client_tmps = client_tmps
        self.neighbors_estimates = neighbors_estimates

        # "AdaG"
        # self.estimate_gossip_error = estimate_gossip_error
        # self.current_weights = current_weights
        # self.m_hat = m_hat

        "BEER"
        self.neighbor_H = neighbor_H
        self.neighbor_G = neighbor_G
        self.H = H
        self.G = G
        self.V = []
        self.previous_gradients = []

        "DeCoM"
        self.gradients = []
        self.gradients_tmp = []
        self.client_theta_hat = client_weights  # initial is model weights
        self.neighbors_theta_hat = neighbor_models  # initials are model weights
        self.client_g_hat = client_accumulates  # initials are zero
        self.neighbors_g_hat = neighbors_accumulates  # initials are zeros
        self.v = []  # gradient estimate
        self.previous_V = []
        self.previous_X = client_weights

        "CEDAS"
        # self.trained_weights = client_accumulates
        self.diffusion = client_accumulates  # zeros
        self.h = []  # initial model weights
        self.hw = []  # h_omega
        # self.y_hat_plus = client_accumulates
        # self.y = client_accumulates  # zeros
        self.updates = neighbors_accumulates

        "MOTEF"
        # self.client_H = client_accumulates
        # self.neighbors_H = neighbors_accumulates
        # self.client_G = client_accumulates
        # self.neighbors_G = neighbors_accumulates
        # self.V = []
        self.M = []
        # self.M = client_accumulates
        self.previous_M = client_accumulates

        "Adaptive gamma in DEFD"
        self.control = control
        self.adaptive = adaptive
        self.threshold = threshold
        self.org_gamma = self.client_compressor[0].discount_parameter
        self.old_error = [torch.zeros_like(self.client_weights[n]) for n in range(self.num_clients)]

        "Testing parameter"
        self.Alpha = []
        self.alpha_max = alpha_max
        self.compression_method = compression_method
        self.changes_ratio = []
        "Debugging parameters"
        self.max = []
        self.change_iter_num = []
        self.error_mag = []
        self.error_ratio = []
        self.logger()
        self.gamma = 1
        # self.coefficient = torch.ones_like(self.models[0])

    def logger(self):
        print(' compression method:', self.compression_method, '\n',
              'apply control method: ', self.control, '\n',
              'alpha upper bound in theory: ', self.alpha_max, '\n',
              'running algorithm: ', self.algorithm_name, '\n')

    def _training(self, data_loader, client_weights, model):
        model.assign_weights(weights=client_weights)
        model.model.train()
        for i in range(self.local_iter):
            # images, labels = next(iter(data_loader))
            images, labels = data_loader
            images, labels = images.to(self.device), labels.to(self.device)
            if self.data_transform is not None:
                images = self.data_transform(images)

            model.optimizer.zero_grad()
            pred = model.model(images)
            # print(pred, len(pred))
            loss = model.loss_function(pred, labels)
            loss.backward()
            model.optimizer.step()

        # grads = [param.grad.view(-1) for param in model.model.parameters() if param.grad is not None]
        # print(grads)
        # all_grads = torch.cat(grads)
        # print(all_grads)
        # gradient_variance = torch.var(all_grads)
        # print(gradient_variance, np.sqrt(gradient_variance))

        trained_model = model.get_weights()  # x_t - \eta * gradients
        return trained_model

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

    def EFD_dc(self, iter_num, normalization):
        # Averaged_weights = self._average_updates_EFD(updates=self.neighbor_models, update=self.client_weights)  # X_t(W-I)
        Averaged_weights = self._average_updates(updates=self.neighbor_models)
        alpha = []
        maxes = []
        error_mag_i = []
        error_ratio_i = []

        # if iter_num % 200 == 0:
        #     self.threshold = self.threshold - 0.1
        learning_rate = self.models[0].learning_rate

        for n in range(self.num_clients):
            # print(iter_num, self.models[n].learning_rate)
            if self.control:
                qt = self.client_partition[n].get_q(iter_num)
                if np.random.binomial(1, qt) == 1:
                    Vector_update = self._training(data_loader=self.data_loaders[n],
                                                   client_weights=self.client_weights[n], model=self.models[n])
                    Vector_update -= self.client_weights[n]  # gradient
                    Vector_update += Averaged_weights[n]
                else:
                    Vector_update = Averaged_weights[n]
                pass
            else:
                images, labels = next(iter(self.data_loaders[n]))
                Vector_update = self._training(data_loader=[images, labels],
                                               client_weights=self.client_weights[n], model=self.models[n])
                Vector_update -= self.client_weights[n]  # gradient
                gradient = Vector_update
                gradient_norm = torch.sum(torch.square(Vector_update)).item()
                gradient_and_error_norm = torch.sum(torch.square(Vector_update + self.client_residuals[n])).item()
                pure_gradient_norm = torch.sum(torch.square(Vector_update / learning_rate)).item()
                # gradient_and_derror_norm = torch.sum(torch.square(Vector_update + self.client_compressor[n].discount_parameter * self.client_residuals[n])).item()
                Vector_update += Averaged_weights[n]

            Vector_update -= self.client_weights[n]  # Difference between averaged weights and local weights

            # print(iter_num, n, torch.sum(torch.square(Vector_update)).item())
            gradient_plus_average_model_norm = torch.sum(torch.square(Vector_update)).item()
            if iter_num == 0:
                error_norm = 1
            else:
                error_norm = torch.sum(torch.square(self.client_residuals[n])).item()
            old_error = self.client_residuals[n]
            error_mag_i.append(error_norm)
            discounted_error_norm = (self.client_compressor[n].discount_parameter)**2 * error_norm

            gradient_plus_average_model_error_norm = gradient_plus_average_model_norm / error_norm
            gradient_plus_average_model_discounted_error_norm = gradient_plus_average_model_norm / discounted_error_norm
            bt_norm = torch.sum(torch.square(Vector_update + self.client_compressor[n].discount_parameter * self.client_residuals[n])).item()
            direct_norm = torch.sum(torch.square(Vector_update + self.client_residuals[n])).item()
            # old_error_norm = torch.sum(torch.square(self.old_error[n])).item()
            # discounted_old_error_norm = (self.client_compressor[n].discount_parameter**2) * old_error_norm
            beta = 0.9
            epsilon = 0.000000000001  # noise: make sure not divide or multiple with zero.

            momentum_error = beta * self.client_residuals[n] + (1 - beta) * self.old_error[n]
            momentum_error_norm = torch.sum(torch.square(momentum_error)).item()
            discounted_momentum_error_norm = torch.sum(torch.square(self.client_compressor[n].discount_parameter * momentum_error)).item()
            difference_error_norm = torch.sum(torch.square(self.client_residuals[n] - self.old_error[n])).item()
            difference_derror_norm = torch.sum(torch.square(self.client_residuals[n] - (self.client_compressor[n].discount_parameter**2) * self.old_error[n])).item()

            if iter_num == 0 or 1:
                coefficient = 1
            else:
                # coefficient = np.sqrt(old_error_norm / error_norm)
                coefficient = np.sqrt(discounted_old_error_norm / error_norm)

            # print('iteration: ', iter_num, 'client: ', n, 'difference norm: ', difference_error_norm, 'error norm: ', error_norm,
            #       'discounted error norm:', discounted_error_norm, 'gradient norm:', gradient_norm, 'whole update norm:', gradient_plus_average_model_norm)
            # print(iter_num, n, gradient_norm / discounted_error_norm)
            self.old_error[n] = self.client_residuals[n]
            self.client_residuals[n] = momentum_error
            "Pre-adjustment"
            if self.adaptive is True:
                self.client_compressor[n].discount_parameter = min(np.sqrt(gradient_norm / (normalization * discounted_momentum_error_norm + epsilon)), 1)  # works well for topk
                # self.client_compressor[n].discount_parameter = min(np.sqrt(gradient_and_error_norm / (normalization * discounted_error_norm)), 1)
                # self.client_compressor[n].discount_parameter = min(np.sqrt(discounted_old_error_norm / error_norm), 1)  # works well for topk
                # self.client_compressor[n].discount_parameter = min(np.sqrt(gradient_plus_average_model_norm / (normalization * momentum_error_norm + epsilon)), 1)  # works well for quantization
                # self.client_compressor[n].discount_parameter = min(np.sqrt(gradient_norm / discounted_error_norm), 1)  # equals to first one
                # self.client_compressor[n].discount_parameter = min(np.sqrt(gradient_plus_average_model_norm / discounted_error_norm), 1)
            "Compression Operator"
            Vector_update, self.client_residuals[n] = self.client_compressor[n].get_trans_bits_and_residual(iter=iter_num, w_tmp=Vector_update, w_residual=self.client_residuals[n], device=self.device, neighbors=self.neighbors[n])

            Update_norm = torch.sum(torch.square(Vector_update)).item()
            new_error_norm = torch.sum(torch.square(self.client_residuals[n])).item()

            if iter_num == 0:
                new_to_old_error_ratio = 1
                self.initial_error_norm = new_error_norm
            else:
                new_to_old_error_ratio = discounted_error_norm / new_error_norm

            # print(iter_num, n, pure_gradient_norm, discounted_error_norm, bt_norm, Update_norm, new_error_norm, new_to_old_error_ratio)
            # new_error_to_bt = new_error_norm / bt_norm
            # error_to_bt = error_norm / bt_norm
            gradient_to_error = gradient_norm / error_norm
            # print(iter_num, n, new_error_to_bt, error_to_bt)
            # print(iter_num, n, np.sqrt(gradient_to_error))
            alpha = 0.9
            # difference_error_norm = torch.sum(torch.square(self.client_residuals[n] - old_error)).item()
            # momentum_norm = alpha * error_norm + (1-alpha) * new_error_norm
            # print(iter_num, n, new_error_norm, np.sqrt(new_error_norm))
            # print(iter_num, n, momentum_norm, np.sqrt(momentum_norm))
            # print(iter_num, n, difference_error_norm, np.sqrt(difference_error_norm))
            # print(iter_num, n, momentum_norm / new_error_norm, np.sqrt(momentum_norm / new_error_norm), np.sqrt(difference_error_norm / new_error_norm))

            # error_ratio_i.append(error_to_bt)
            # beta = new_error_norm / bt_norm

            # sigma = 0.11
            # a = np.sqrt(n)
            # if iter_num > 0:
            #     print(iter_num, n, 'e(t): ', error_norm, 'beta: ', beta, '1/beta: ', learning_rate/beta,
            #           'error ratio: ', new_to_old_error_ratio,
            #           'error ratio sqrt: ', np.sqrt(new_to_old_error_ratio),
            #           'Ada_org: ', learning_rate * np.sqrt(1/learning_rate) / np.sqrt(new_error_norm),
            #           'new: ', learning_rate / (np.sqrt(new_error_norm) * np.sqrt(self.num_clients)))

            "post adjustment according to sqrt of error norm"
            # sigma = 1
            sigma = 1 / np.sqrt(learning_rate)
            # sigma = np.sqrt(self.num_clients)
            # if self.adaptive is True:
            #     self.client_compressor[n].discount_parameter = min(np.sqrt(discounted_error_norm / new_error_norm), 1)
            #     if self.compression_method == 'topk':
            #         self.client_compressor[n].discount_parameter = min(np.sqrt(discounted_error_norm / new_error_norm), 1)
            #     elif self.compression_method == 'quantization':
            #         self.client_compressor[n].discount_parameter = min(np.sqrt(gradient_plus_average_model_norm / discounted_error_norm), 1)
                # if self.compression_method == 'topk':
                #     self.client_compressor[n].discount_parameter = min((self.org_gamma * learning_rate * sigma) / (np.sqrt(new_error_norm) + epsilon), 1)
                # elif self.compression_method == 'quantization':
                #     self.client_compressor[n].discount_parameter = min((self.org_gamma * learning_rate * sigma) / (np.sqrt(new_error_norm) * normalization + epsilon), 1)

            self.client_weights[n] += Vector_update
            for m in range(self.num_clients):
                if n in self.neighbors[m]:
                    self.neighbor_models[m][self.neighbors[m].index(n)] += Vector_update
        # self.Alpha.append(sum(alpha)/len(alpha))  # error ratio
        # print(iter_num, alpha, '\n')
        # self.error_mag.append(sum(error_mag_i) / len(error_mag_i))
        # self.error_ratio.append(sum(error_ratio_i) / len(error_ratio_i))

        print(iter_num, [self.client_compressor[n].discount_parameter for n in range(self.num_clients)], '\n')

    def DCD(self, iter_num):
        Averaged_weights = self._average_updates(updates=self.neighbor_models)

        for n in range(self.num_clients):
            if self.control:
                # qt = self.client_partition[n].get_q(iter_num)
                # if np.random.binomial(1, qt) == 1:
                #     Vector_update = self._training(data_loader=self.data_loaders[n],
                #                                    client_weights=self.client_weights[n],
                #                                    model=self.models[n])
                #     Vector_update -= self.client_weights[n]  # gradient
                #     Vector_update += Averaged_weights[n]
                # else:
                #     Vector_update = Averaged_weights[n]
                pass
            else:
                images, labels = next(iter(self.data_loaders[n]))
                Vector_update = self._training(data_loader=[images, labels],
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

    def _averaged_choco(self, updates, update):
        Averaged = []
        for i in range(self.num_clients):
            summation = torch.zeros_like(update[0])
            for j in range(len(updates[i])):
                summation += (1/len(updates[i])) * (updates[i][j] - update[i])
            Averaged.append(summation)
        return Averaged

    # def CHOCO(self, iter_num, consensus):  # 1
    #     Averaged_accumulate = self._averaged_choco(updates=self.neighbor_accumulates, update=self.client_accumulates)
    #
    #     for n in range(self.num_clients):
    #         self.client_weights[n] = self.client_tmps[n] + consensus * Averaged_accumulate[n]
    #         self.client_tmps[n] = self._training(data_loader=self.data_loaders[n], client_weights=self.client_weights[n], model=self.models[n])
    #
    #         Vector_update = self.client_weights[n] - self.client_accumulates[n]
    #         Vector_update, _ = self.client_compressor[n].get_trans_bits_and_residual(w_tmp=Vector_update, iter=iter_num,
    #                                                                                  w_residual=self.client_residuals[n],
    #                                                                                  device=self.device, neighbors=self.neighbors[n])
    #         self.client_accumulates[n] += Vector_update  # Vector Update is q_t
    #         for m in range(self.num_clients):
    #             if n in self.neighbors[m]:
    #                 self.neighbor_accumulates[m][self.neighbors[m].index(n)] += Vector_update

    def CHOCO(self, iter_num, consensus):
        for n in range(self.num_clients):
            images, labels = next(iter(self.data_loaders[n]))
            self.client_tmps[n] = self._training(data_loader=[images, labels], client_weights=self.client_weights[n], model=self.models[n])

            Vector_update = self.client_weights[n] - self.client_accumulates[n]
            Vector_update, _ = self.client_compressor[n].get_trans_bits_and_residual(w_tmp=Vector_update, iter=iter_num,
                                                                                     w_residual=self.client_residuals[n],
                                                                                     device=self.device, neighbors=self.neighbors[n])
            self.client_accumulates[n] += Vector_update  # Vector Update is q_t
            for m in range(self.num_clients):
                if n in self.neighbors[m]:
                    self.neighbor_accumulates[m][self.neighbors[m].index(n)] += Vector_update

        Averaged_accumulate = self._averaged_choco(updates=self.neighbor_accumulates, update=self.client_accumulates)

        for n in range(self.num_clients):
            self.client_weights[n] = self.client_tmps[n] + consensus * Averaged_accumulate[n]

    def BEER(self, iter_num, gamma, learning_rate):
        weighted_H = self._averaged_choco(updates=self.neighbor_H, update=self.H)
        weighted_G = self._averaged_choco(updates=self.neighbor_G, update=self.G)

        for n in range(self.num_clients):
            if iter_num == 0:
                images, labels = next(iter(self.data_loaders[n]))
                training_weights = self._training(data_loader=[images, labels], client_weights=self.client_weights[n], model=self.models[n])
                initial_gradients = (self.client_weights[n] - training_weights) / learning_rate
                # initial_gradients = self.client_weights[n] - training_weights
                self.V.append(initial_gradients)
                self.previous_gradients.append(initial_gradients)

            self.client_weights[n] = self.client_weights[n] + gamma * weighted_H[n] - learning_rate * self.V[n]
            H_update = self.client_weights[n] - self.H[n]
            H_update, _ = self.client_compressor[n].get_trans_bits_and_residual(w_tmp=H_update, iter=iter_num,
                                                                                     w_residual=self.client_residuals[n],
                                                                                     device=self.device,
                                                                                     neighbors=self.neighbors[n])
            self.H[n] += H_update
            for m in range(self.num_clients):
                if n in self.neighbors[m]:
                    self.neighbor_H[m][self.neighbors[m].index(n)] += H_update

            images, labels = next(iter(self.data_loaders[n]))
            next_train_weights = self._training(data_loader=[images, labels], client_weights=self.client_weights[n], model=self.models[n])
            next_gradients = (self.client_weights[n] - next_train_weights) / learning_rate
            # next_gradients = self.client_weights[n] - next_train_weights

            self.V[n] = self.V[n] + gamma * weighted_G[n] + next_gradients - self.previous_gradients[n]
            self.previous_gradients[n] = next_gradients

            G_update = self.V[n] - self.G[n]
            G_update, _ = self.client_compressor[n].get_trans_bits_and_residual(w_tmp=G_update, iter=iter_num,
                                                                                w_residual=self.client_residuals[n],
                                                                                device=self.device,
                                                                                neighbors=self.neighbors[n])
            self.G[n] += G_update
            for m in range(self.num_clients):
                if n in self.neighbors[m]:
                    self.neighbor_G[m][self.neighbors[m].index(n)] += G_update

    # def DeCoM(self, iter_num, gamma, learning_rate, beta):  # Have problem with Quantization compression
    #     for n in range(self.num_clients):
    #         if iter_num == 0:
    #             images, labels = next(iter(self.data_loaders[n]))
    #             training_weights = self._training(data_loader=[images, labels], client_weights=self.client_weights[n], model=self.models[n])
    #             initial_gradients = (self.client_weights[n] - training_weights) / learning_rate
    #             self.v.append(initial_gradients)
    #             # self.previous_V.append(initial_gradients)
    #             self.gradients.append(initial_gradients)
    #             # self.gradients_tmp.append(initial_gradients)
    #
    #         images, labels = next(iter(self.data_loaders[n]))
    #         previous_grad = self._training(data_loader=[images, labels], client_weights=self.previous_X[n], model=self.models[n])
    #         previous_grad = (self.previous_X[n] - previous_grad) / learning_rate
    #         current_grad = self._training(data_loader=[images, labels], client_weights=self.client_weights[n], model=self.models[n])
    #         current_grad = (self.client_weights[n] - current_grad) / learning_rate
    #
    #         self.previous_V[n] = self.v[n]
    #         self.v[n] = current_grad + (1 - beta) * (self.v[n] - previous_grad)
    #
    #         g_update = self.gradients[n] + self.v[n] - self.previous_V[n] - self.client_g_hat[n]
    #         # g_update, _ = self.client_compressor[n].get_trans_bits_and_residual(w_tmp=g_update, iter=iter_num,
    #         #                                                                     w_residual=self.client_residuals[n],
    #         #                                                                     device=self.device,
    #         #                                                                     neighbors=self.neighbors[n])
    #         self.client_g_hat[n] += g_update
    #         for m in range(self.num_clients):
    #             if n in self.neighbors[m]:
    #                 self.neighbors_g_hat[m][self.neighbors[m].index(n)] += g_update
    #
    #     weighted_g = self._averaged_choco(updates=self.neighbors_g_hat, update=self.client_g_hat)
    #     for n in range(self.num_clients):
    #
    #         self.gradients[n] += self.v[n] - self.previous_V[n] + gamma * weighted_g[n]
    #
    #         self.previous_X[n] = self.client_weights[n]
    #
    #         theta_update = self.client_weights[n] - learning_rate * self.gradients[n] - self.client_theta_hat[n]
    #         # theta_update, _ = self.client_compressor[n].get_trans_bits_and_residual(w_tmp=theta_update, iter=iter_num,
    #         #                                                                         w_residual=self.client_residuals[n],
    #         #                                                                         device=self.device,
    #         #                                                                         neighbors=self.neighbors[n])
    #
    #         self.client_theta_hat[n] += theta_update
    #         for m in range(self.num_clients):
    #             if n in self.neighbors[m]:
    #                 self.neighbors_theta_hat[m][self.neighbors[m].index(n)] += theta_update
    #
    #     weighted_theta = self._averaged_choco(updates=self.neighbors_theta_hat, update=self.client_theta_hat)
    #     for n in range(self.num_clients):
    #         self.client_weights[n] += gamma * weighted_theta[n] - learning_rate * self.gradients[n]

    def DeCoM(self, iter_num, gamma, learning_rate, beta):  # Have problem with Quantization compression
        # s = 32
        for n in range(self.num_clients):
            if iter_num == 0:
                images, labels = next(iter(self.data_loaders[n]))
                training_weights = self._training(data_loader=[images, labels], client_weights=self.client_weights[n],
                                                  model=self.models[n])
                initial_gradients = (self.client_weights[n] - training_weights) / learning_rate
                self.v.append(initial_gradients)
                self.previous_V.append(initial_gradients)
                self.gradients.append(initial_gradients)
                self.gradients_tmp.append(initial_gradients)

            'client_weights --- theta'
            self.client_tmps[n] = self.client_weights[n] - learning_rate * self.gradients[n]
            theta_update = self.client_tmps[n] - self.client_theta_hat[n]
            theta_update, _ = self.client_compressor[n].get_trans_bits_and_residual(w_tmp=theta_update, iter=iter_num,
                                                                                    w_residual=self.client_residuals[n],
                                                                                    device=self.device,
                                                                                    neighbors=self.neighbors[n])
            # theta_update = random_quantize_mat(theta_update, s=s)[0]
            self.client_theta_hat[n] += theta_update
            for m in range(self.num_clients):
                if n in self.neighbors[m]:
                    self.neighbors_theta_hat[m][self.neighbors[m].index(n)] += theta_update

        weighted_theta = self._averaged_choco(updates=self.neighbors_theta_hat, update=self.client_theta_hat)
        for n in range(self.num_clients):
            # next_dataloader = copy.deepcopy(self.data_loaders[n])
            images, labels = next(iter(self.data_loaders[n]))
            f_hat_current = self._training(data_loader=[images, labels], client_weights=self.client_weights[n],
                                           model=self.models[n])
            f_hat_current = (self.client_weights[n] - f_hat_current) / learning_rate

            self.client_weights[n] = self.client_tmps[n] + gamma * weighted_theta[n]
            f_hat_next = self._training(data_loader=[images, labels], client_weights=self.client_weights[n],
                                        model=self.models[n])
            f_hat_next = (self.client_weights[n] - f_hat_next) / learning_rate

            self.v[n] = beta * f_hat_next + (1 - beta) * (self.v[n] + f_hat_next - f_hat_current)
            # self.v[n] = f_hat_next + (1 - beta) * (self.v[n] - f_hat_current)

            self.gradients_tmp[n] = self.gradients[n] + self.v[n] - self.previous_V[n]
            self.previous_V[n] = self.v[n]

            g_update = self.gradients_tmp[n] - self.client_g_hat[n]
            g_update, _ = self.client_compressor[n].get_trans_bits_and_residual(w_tmp=g_update, iter=iter_num,
                                                                                w_residual=self.client_residuals[n],
                                                                                device=self.device,
                                                                                neighbors=self.neighbors[n])
            # g_update = random_quantize_mat(theta_update, s=s)[0]
            self.client_g_hat[n] += g_update
            for m in range(self.num_clients):
                if n in self.neighbors[m]:
                    self.neighbors_g_hat[m][self.neighbors[m].index(n)] += g_update

        weighted_g = self._averaged_choco(updates=self.neighbors_g_hat, update=self.client_g_hat)
        for n in range(self.num_clients):
            self.gradients[n] = self.gradients_tmp[n] + gamma * weighted_g[n]

    def CEDAS(self, iter_num, alpha, gamma):
        Trained_weights = []
        Y_hat_plus = []
        for n in range(self.num_clients):
            if iter_num == 0:
                self.h.append(self.client_weights[n])
                self.hw.append(self.client_weights[n])
                images, labels = next(iter(self.data_loaders[n]))
                self.client_weights[n] = self._training(data_loader=[images, labels],
                                             client_weights=self.client_weights[n],
                                             model=self.models[n])

            images, labels = next(iter(self.data_loaders[n]))
            trained_weights = self._training(data_loader=[images, labels],
                                             client_weights=self.client_weights[n],
                                             model=self.models[n])

            Trained_weights.append(trained_weights)
            y = trained_weights - self.diffusion[n]
            "COMM start"
            q = y - self.h[n]
            q, _ = self.client_compressor[n].get_trans_bits_and_residual(w_tmp=q, iter=iter_num,
                                                                         w_residual=self.client_residuals[n],
                                                                         device=self.device,
                                                                         neighbors=self.neighbors[n])
            y_hat_plus = self.h[n] + q
            Y_hat_plus.append(y_hat_plus)
            for m in range(self.num_clients):
                if n in self.neighbors[m]:
                    self.updates[m][self.neighbors[m].index(n)] = q

        Averaged_updates = self._average_updates(updates=self.updates)

        for n in range(self.num_clients):
            yw_hat_plus = self.hw[n] + Averaged_updates[n]

            self.h[n] = (1-alpha) * self.h[n] + alpha * Y_hat_plus[n]
            self.hw[n] = (1-alpha) * self.hw[n] + alpha * yw_hat_plus  # here
            "COMM end"

            self.diffusion[n] += (gamma / 2) * (Y_hat_plus[n] - yw_hat_plus)
            self.client_weights[n] = Trained_weights[n] - self.diffusion[n]
            # self.client_weights[n] = Trained_weights[n]

    "MOTEF and MOTEF_VR require large batch size?"
    def MoTEF(self, iter_num, gamma, learning_rate, Lambda):  # Binary classification?
        weighted_H = self._averaged_choco(updates=self.neighbor_H, update=self.H)
        weighted_G = self._averaged_choco(updates=self.neighbor_G, update=self.G)
        for n in range(self.num_clients):
            if iter_num == 0:
                images, labels = next(iter(self.data_loaders[n]))
                training_weights = self._training(data_loader=[images, labels], client_weights=self.client_weights[n], model=self.models[n])
                initial_gradients = (self.client_weights[n] - training_weights) / learning_rate
                # initial_gradients = self.client_weights[n] - training_weights
                self.V.append(initial_gradients)
                self.M.append(initial_gradients)
                # self.M.append(torch.zeros_like(initial_gradients))

            self.client_weights[n] += gamma * weighted_H[n] - learning_rate * self.V[n]
            Q_h_update = self.client_weights[n] - self.H[n]
            Q_h_update, _ = self.client_compressor[n].get_trans_bits_and_residual(w_tmp=Q_h_update, iter=iter_num,
                                                                                 w_residual=self.client_residuals[n],
                                                                                 device=self.device,
                                                                                 neighbors=self.neighbors[n])
            self.H[n] += Q_h_update
            for m in range(self.num_clients):
                if n in self.neighbors[m]:
                    self.neighbor_H[m][self.neighbors[m].index(n)] += Q_h_update

            # print(self.previous_M[n], self.M)
            self.previous_M[n] = copy.deepcopy(self.M[n])

            images, labels = next(iter(self.data_loaders[n]))
            trained_weights = self._training(data_loader=[images, labels],
                                             client_weights=self.client_weights[n],
                                             model=self.models[n])
            gradients = (self.client_weights[n] - trained_weights) / learning_rate

            self.M[n] = (1 - Lambda) * self.M[n] + Lambda * gradients
            self.V[n] += gamma * weighted_G[n] + self.M[n] - self.previous_M[n]

            Q_g_update = self.V[n] - self.G[n]
            Q_g_update, _ = self.client_compressor[n].get_trans_bits_and_residual(w_tmp=Q_g_update, iter=iter_num,
                                                                                  w_residual=self.client_residuals[n],
                                                                                  device=self.device,
                                                                                  neighbors=self.neighbors[n])

            self.G[n] += Q_g_update
            for m in range(self.num_clients):
                if n in self.neighbors[m]:
                    self.neighbor_G[m][self.neighbors[m].index(n)] += Q_g_update

    def MOTEF_VR(self, iter_num, gamma, learning_rate, Lambda):
        weighted_H = self._averaged_choco(updates=self.neighbor_H, update=self.H)
        weighted_G = self._averaged_choco(updates=self.neighbor_G, update=self.G)
        for n in range(self.num_clients):
            if iter_num == 0:
                images, labels = next(iter(self.data_loaders[n]))
                training_weights = self._training(data_loader=[images, labels],
                                                  client_weights=self.client_weights[n], model=self.models[n])
                initial_gradients = (self.client_weights[n] - training_weights) / learning_rate
                # initial_gradients = self.client_weights[n] - training_weights
                self.V.append(initial_gradients)
                self.M.append(initial_gradients)
                # self.M.append(torch.zeros_like(initial_gradients))

            images, labels = next(iter(self.data_loaders[n]))
            trained_weights_current = self._training(data_loader=[images, labels],
                                             client_weights=self.client_weights[n],
                                             model=self.models[n])
            gradients_current = (self.client_weights[n] - trained_weights_current) / learning_rate

            self.client_weights[n] += gamma * weighted_H[n] - learning_rate * self.V[n]
            Q_h_update = self.client_weights[n] - self.H[n]
            Q_h_update, _ = self.client_compressor[n].get_trans_bits_and_residual(w_tmp=Q_h_update, iter=iter_num,
                                                                                 w_residual=self.client_residuals[n],
                                                                                 device=self.device,
                                                                                 neighbors=self.neighbors[n])
            self.H[n] += Q_h_update
            for m in range(self.num_clients):
                if n in self.neighbors[m]:
                    self.neighbor_H[m][self.neighbors[m].index(n)] += Q_h_update

            images, labels = next(iter(self.data_loaders[n]))
            trained_weights_next = self._training(data_loader=[images, labels],
                                             client_weights=self.client_weights[n],
                                             model=self.models[n])
            gradients_next = (self.client_weights[n] - trained_weights_next) / learning_rate

            self.previous_M[n] = self.M[n]
            self.M[n] = gradients_next + (1 - Lambda) * (self.M[n] - gradients_current)
            self.V[n] += gamma * weighted_G[n] + self.M[n] - self.previous_M[n]

            Q_g_update = self.V[n] - self.G[n]
            Q_g_update, _ = self.client_compressor[n].get_trans_bits_and_residual(w_tmp=Q_g_update, iter=iter_num,
                                                                                  w_residual=self.client_residuals[n],
                                                                                  device=self.device,
                                                                                  neighbors=self.neighbors[n])

            self.G[n] += Q_g_update
            for m in range(self.num_clients):
                if n in self.neighbors[m]:
                    self.neighbor_G[m][self.neighbors[m].index(n)] += Q_g_update

    # def AdaG_SGD(self, iter_num, beta, consensus, epsilon):  # x_hat_0 = 0 / u_i_0 = 0 U_t_i = error / AdaG needs very small consensus (gamma)
    #     # The original algorithm possibly has errors of the update rule in the paper, not consistent with code
    #
    #     for n in range(self.num_clients):
    #         self.client_tmps[n] = self._training(data_loader=self.data_loaders[n], client_weights=self.client_weights[n], model=self.models[n])
    #
    #         Vector_update = self.client_tmps[n] - self.client_accumulates[n]
    #         Vector_update, _ = self.client_compressor[n].get_trans_bits_and_residual(w_tmp=Vector_update, iter=iter_num,
    #                                                                                  w_residual=self.client_residuals[n],
    #                                                                                  device=self.device,
    #                                                                                  neighbors=self.neighbors[n])
    #         self.client_accumulates[n] += Vector_update  # Vector Update is q_t
    #         for m in range(self.num_clients):
    #             if n in self.neighbors[m]:
    #                 self.neighbor_accumulates[m][self.neighbors[m].index(n)] += Vector_update
    #
    #     Averaged_accumulate = self._averaged_choco(updates=self.neighbor_accumulates, update=self.client_accumulates)
    #
    #     for l in range(self.num_clients):
    #         self.estimate_gossip_error[l] = (beta * self.estimate_gossip_error[l]) + ((1-beta) * torch.square(Averaged_accumulate[l]))
    #
    #         # adapt_factor = torch.sqrt(self.estimate_gossip_error[l]) + epsilon
    #         adapt_factor = torch.sqrt(self.estimate_gossip_error[l] / (1-(beta**iter_num))) + epsilon
    #         adapted_consensus = consensus / adapt_factor
    #         adapted_consensus = torch.clamp(adapted_consensus, max=1.0)  # Problem: all ones, needs very small consensus
    #
    #         self.client_weights[l] = self.client_tmps[l] + adapted_consensus * Averaged_accumulate[l]
    #         # self.client_weights[l] = self.client_weights[l] + adapted_consensus * Averaged_accumulate[l]  # Update rule in paper, not working
    #
    # def Comp_QSADDLe(self, iter_num, rho, beta, learning_rate, consensus, mu):  # Cannot work with large learning rate
    #
    #     self.current_weights = self.client_weights
    #     for n in range(self.num_clients):
    #         trained_weights = self._training(data_loader=self.data_loaders[n], client_weights=self.client_weights[n], model=self.models[n])
    #         gradients = -(trained_weights - self.client_weights[n]) / learning_rate
    #
    #         zeta = rho * (gradients / torch.norm(gradients))
    #         trained_weights_1 = self._training(data_loader=self.data_loaders[n], client_weights=self.client_weights[n]+zeta, model=self.models[n])
    #         gradients_1 = -(trained_weights_1 - (self.client_weights[n]+zeta)) / learning_rate
    #
    #         # m_t = beta * self.m_hat[n] + gradients
    #         m_t = beta * self.m_hat[n] + gradients_1
    #
    #         self.client_tmps[n] = self.client_weights[n] - learning_rate * m_t
    #
    #     Averaged_accumulate = self._averaged_choco(updates=self.neighbor_accumulates, update=self.client_accumulates)
    #
    #     for n in range(self.num_clients):
    #         self.client_weights[n] = self.client_tmps[n] + consensus * Averaged_accumulate[n]
    #
    #         distance = self.current_weights[n] - self.client_weights[n]
    #         distance = distance / learning_rate  # actually is gradients
    #
    #         self.m_hat[n] = mu * self.m_hat[n] + (1-mu) * distance
    #         Vector_update = self.client_weights[n] - self.client_accumulates[n]
    #         Vector_update, _ = self.client_compressor[n].get_trans_bits_and_residual(w_tmp=Vector_update, iter=iter_num,
    #                                                                                  w_residual=self.client_residuals[n],
    #                                                                                  device=self.device,
    #                                                                                  neighbors=self.neighbors[n])
    #
    #         self.client_accumulates[n] += Vector_update
    #         for m in range(self.num_clients):
    #             if n in self.neighbors[m]:
    #                 self.neighbor_accumulates[m][self.neighbors[m].index(n)] += Vector_update
    #
    # def ECD(self, iter_num):
    #     Averaged_weights = self._average_updates(updates=self.neighbors_estimates)
    #
    #     for n in range(self.num_clients):
    #         current_weights = self.client_weights[n]
    #         if self.control:
    #             qt = self.client_partition[n].get_q(iter_num)
    #             if np.random.binomial(1, qt) == 1:
    #                 Vector_update = self._training(data_loader=self.data_loaders[n],
    #                                                client_weights=self.client_weights[n],
    #                                                model=self.models[n])
    #                 Vector_update -= self.client_weights[n]  # gradient
    #                 Vector_update += Averaged_weights[n]
    #             else:
    #                 Vector_update = Averaged_weights[n]
    #         else:
    #             Vector_update = self._training(data_loader=self.data_loaders[n],
    #                                            client_weights=self.client_weights[n],
    #                                            model=self.models[n])
    #             Vector_update -= self.client_weights[n]  # gradient
    #             Vector_update += Averaged_weights[n]
    #
    #         self.client_weights[n] = Vector_update
    #
    #         z_vector = (1 - 0.5 * iter_num) * current_weights + 0.5 * iter_num * Vector_update
    #         z_vector, _ = self.client_compressor[n].get_trans_bits_and_residual(iter=iter_num, w_tmp=z_vector,
    #                                                                             w_residual=self.client_residuals[n],
    #                                                                             device=self.device, neighbors=self.neighbors[n])
    #         for m in range(self.num_clients):
    #             if n in self.neighbors[m]:
    #                 self.neighbors_estimates[m][self.neighbors[m].index(n)] = (1 - (2/iter_num)) * self.neighbors_estimates[m][self.neighbors[m].index(n)] + (2/iter_num) * z_vector
    #







