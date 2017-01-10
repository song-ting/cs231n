import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.

    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    N = X.shape[0]
    class_input = X.dot(W)  # (N, C)
    class_prob = softmax(class_input)  # (N, C)
    data_loss = np.sum(-np.log(class_prob[np.arange(N), y])) / N
    reg_loss = 0.5 * reg * np.sum(W ** 2)
    loss = data_loss + reg_loss

    dclass_input = class_prob.copy()  # (N, C)
    dclass_input[np.arange(N), y] -= 1.0
    dclass_input /= N
    dW = X.T.dot(dclass_input) + reg * W  # (D, C)
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    N = X.shape[0]
    class_input = X.dot(W)  # (N, C)
    class_prob = softmax(class_input)  # (N, C)
    data_loss = np.sum(-np.log(class_prob[np.arange(N), y])) / N
    reg_loss = 0.5 * reg * np.sum(W ** 2)
    loss = data_loss + reg_loss

    dclass_input = class_prob.copy()  # (N, C)
    dclass_input[np.arange(N), y] -= 1.0
    dclass_input /= N
    dW = X.T.dot(dclass_input) + reg * W  # (D, C)
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax(Z):
    Z_max = np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z - Z_max)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
