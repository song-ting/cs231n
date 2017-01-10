from cs231n.classifiers.cnn import *
from cs231n import *
import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.cnn import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.solver import Solver

data = get_CIFAR10_data()

filter_size_list = [3, 5]
num_filters_list = [8, 16, 32]
conv_layers_list = [2, 3]
# lr_list = [3, 5, 7] * 1e-4

for filter_size in filter_size_list:
    for num_filters in num_filters_list:
        for conv_layers in conv_layers_list:
            model = ConvNet(conv_relu_pool_layers=conv_layers, fc_layers=1, input_dim=(3, 32, 32),
                            num_filters=num_filters,
                            filter_size=filter_size, hidden_dim=100, num_classes=10, weight_scale=1e-3, dropout=0,
                            use_batchnorm=False, dtype=np.float32, seed=None, reg=1e-4)

            solver = Solver(model, data,
                            num_epochs=10, batch_size=100,
                            update_rule='adam',
                            optim_config={
                                'learning_rate': 8e-4,
                            },
                            verbose=True, print_every=500)
            print filter_size, num_filters, conv_layers
            solver.train()
