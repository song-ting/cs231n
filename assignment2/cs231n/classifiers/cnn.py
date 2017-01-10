import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        (C, H, W) = input_dim
        self.params['W1'] = np.random.normal(0., weight_scale, (num_filters, C, filter_size, filter_size))
        self.params['b1'] = np.zeros(num_filters)
        self.params['W2'] = np.random.normal(0., weight_scale, (num_filters * (H / 2) * (W / 2), hidden_dim))
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = np.random.normal(0., weight_scale, (hidden_dim, num_classes))
        self.params['b3'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        # forward propagation
        conv_relu_pool_out, conv_relu_pool_cache = conv_relu_pool_forward(X, W1, b1, conv_param,
                                                                          pool_param)
        # conv_relu_pool_cache = (conv_cache, relu_cache, pool_cache)

        affine_relu_out, affine_relu_cache = affine_relu_forward(conv_relu_pool_out, W2, b2)
        # affine_relu_cache = (fc_cache, relu_cache)

        scores, affine_cache = affine_forward(affine_relu_out, W3, b3)  # shape (N, num_class)
        # affine_cache = (fc_cache, relu_cache)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        # backward propagation
        reg_loss = 0.
        data_loss, dscores = softmax_loss(scores, y)
        daffine_input, grads['W3'], grads['b3'] = affine_backward(dscores, affine_cache)
        reg_loss += 0.5 * self.reg * np.sum(W3 ** 2)
        grads['W3'] += self.reg * W3

        daffine_relu_input, grads['W2'], grads['b2'] = affine_relu_backward(daffine_input, affine_relu_cache)
        reg_loss += 0.5 * self.reg * np.sum(W2 ** 2)
        grads['W2'] += self.reg * W2

        dX, grads['W1'], grads['b1'] = conv_relu_pool_backward(daffine_relu_input, conv_relu_pool_cache)
        reg_loss += 0.5 * self.reg * np.sum(W1 ** 2)
        grads['W1'] += self.reg * W1

        loss = data_loss + reg_loss
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class ConvNet(object):
    """
    A multi-layer convolutional network with the following architecture:

    {conv - [batch norm] - relu - 2x2 max pool} x N - (affine - [batch norm] - relu) x M - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, conv_relu_pool_layers, fc_layers, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, dropout=0, use_batchnorm=False,
                 reg=0.0, dtype=np.float32, seed=None):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.conv_relu_pool_layers = conv_relu_pool_layers
        self.num_layers = conv_relu_pool_layers + fc_layers + 1
        self.use_dropout = dropout > 0
        self.params = {}
        self.use_batchnorm = use_batchnorm
        self.reg = reg
        self.dtype = dtype

        # Initialize weights and biases for the multi-layer convolutional          #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # secend layer.
        (C, H, W) = input_dim
        for i in xrange(1, self.num_layers + 1):
            if i <= conv_relu_pool_layers:  # {conv - [batch norm] - relu - 2x2 max pool} x N
                if i == 1:
                    self.params['W' + str(i)] = np.random.normal(0., weight_scale,
                                                                 (num_filters, C, filter_size, filter_size))
                else:
                    self.params['W' + str(i)] = np.random.normal(0., weight_scale,
                                                                 (num_filters, num_filters, filter_size, filter_size))
                self.params['b' + str(i)] = np.zeros(num_filters)
            elif i < self.num_layers:  # (affine - [batch norm] - relu) x M
                self.params['W' + str(i)] = np.random.normal(0., weight_scale,
                                                             (num_filters * (H / (2 ** conv_relu_pool_layers) * (
                                                                 W / (2 ** conv_relu_pool_layers))), hidden_dim))
                self.params['b' + str(i)] = np.zeros(hidden_dim)
            elif i == self.num_layers:  # affine - softmax
                self.params['W' + str(i)] = np.random.normal(0., weight_scale, (hidden_dim,
                                                                                num_classes))
                self.params['b' + str(i)] = np.zeros(num_classes)
                break
            if use_batchnorm:  # (N + M) x batch normalization
                self.params['gamma' + str(i)] = np.ones_like(self.params['b' + str(i)])
                self.params['beta' + str(i)] = np.zeros_like(self.params['b' + str(i)])

            # When using dropout we need to pass a dropout_param dictionary to each
            # dropout layer so that the layer knows the dropout probability and the mode
            # (train / test). You can pass the same dropout_param to each dropout layer.
            self.dropout_param = {}
            if self.use_dropout:
                self.dropout_param = {'mode': 'train', 'p': dropout}
                if seed is not None:
                    self.dropout_param['seed'] = seed

            # With batch normalization we need to keep track of running means and
            # variances, so we need to pass a special bn_param object to each batch
            # normalization layer. You should pass self.bn_params[0] to the forward pass
            # of the first batch normalization layer, self.bn_params[1] to the forward
            # pass of the second batch normalization layer, etc.
            self.bn_params = []
            if self.use_batchnorm:
                self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]

            # Cast all parameters to the correct datatype
            for k, v in self.params.iteritems():
                self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param[mode] = mode

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = self.params['W1'].shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        scores = None

        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.
        num_layers = self.num_layers
        out = X
        layer_caches = []  # length num_layers-1
        for i in xrange(1, num_layers + 1):
            if i <= self.conv_relu_pool_layers:  # {conv - [batchnorm] - relu - pool} x N
                if not self.use_batchnorm:  # conv - relu - pool
                    out, conv_relu_pool_cache = conv_relu_pool_forward(out, self.params['W' + str(i)],
                                                                       self.params['b' + str(i)],
                                                                       conv_param, pool_param)
                    layer_caches.append(conv_relu_pool_cache)
                else:  # conv - [batchnorm] - relu - pool
                    out, conv_bn_relu_pool_cache = conv_batchnorm_relu_pool_forward(out, self.params['W' + str(i)],
                                                                                    self.params['b' + str(i)],
                                                                                    conv_param,
                                                                                    self.params['gamma' + str(i)],
                                                                                    self.params['beta' + str(i)],
                                                                                    self.bn_params[i - 1], pool_param)
                    layer_caches.append(conv_bn_relu_pool_cache)

            elif i < num_layers:  # (affine - [batchnorm] - relu) x M
                if not self.use_batchnorm:  # affine relu
                    out, fc_relu_cache = affine_relu_forward(out, self.params['W' + str(i)], self.params['b' + str(i)])
                    layer_caches.append(fc_relu_cache)
                else:  # affine - [batchnorm] - relu
                    out, fc_bn_relu_cache = affine_batchnorm_relu_forward(out, self.params['W' + str(i)],
                                                                          self.params['b' + str(i)],
                                                                          self.params['gamma' + str(i)],
                                                                          self.params['beta' + str(i)],
                                                                          self.bn_params[i - 1])
                    layer_caches.append(fc_bn_relu_cache)

            elif i == num_layers:  # output layer: affine
                scores, fc_cache = affine_forward(out, self.params['W' + str(i)], self.params['b' + str(i)])
                layer_caches.append(fc_cache)

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.

        for i in xrange(num_layers, 0, -1):
            if i == num_layers:  # output layer: affine - softmax
                loss, dscores = softmax_loss(scores, y)
                dout, grads['W' + str(i)], grads['b' + str(i)] = affine_backward(dscores, layer_caches[i - 1])

            elif i > self.conv_relu_pool_layers:  # (affine - [batchnorm] - relu) x M
                if not self.use_batchnorm:  # affine relu
                    dout, grads['W' + str(i)], grads['b' + str(i)] = affine_relu_backward(dout, layer_caches[i - 1])
                elif self.use_batchnorm:  # affine - [batchnorm] - relu
                    dout, grads['W' + str(i)], grads['b' + str(i)], grads['gamma' + str(i)], grads['beta' + str(i)] = \
                        affine_batchnorm_relu_backward(dout, layer_caches[i - 1])

            elif i <= self.conv_relu_pool_layers:  # {conv - [batchnorm] - relu - pool} x N
                if not self.use_batchnorm:  # conv - relu - pool
                    dout, grads['W' + str(i)], grads['b' + str(i)] = conv_relu_pool_backward(dout, layer_caches[i - 1])
                elif self.use_batchnorm:  # conv - [batchnorm] - relu - pool
                    dout, grads['W' + str(i)], grads['b' + str(i)], grads['gamma' + str(i)], grads['beta' + str(i)] = \
                        conv_batchnorm_relu_pool_backward(dout, layer_caches[i - 1])

            loss += 0.5 * self.reg * np.sum(self.params['W' + str(i)] ** 2)
            grads['W' + str(i)] += self.reg * self.params['W' + str(i)]
        return loss, grads
