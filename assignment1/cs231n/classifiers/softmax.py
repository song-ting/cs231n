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
    scores = X.dot(W)  # shape (N, C)
    probs = softmax(scores)  # shape (N, C)
    data_loss = np.sum(-np.log(probs[np.arange(N), y])) / N
    reg_loss = 0.5 * reg * np.sum(W ** 2)
    loss = data_loss + reg_loss

    dscores = probs.copy()  # shape (N, C)
    dscores[np.arange(N), y] -= 1.0
    dscores /= N
    dW = np.dot(X.T, dscores) + reg * W  # shape (D, C)
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
    scores = X.dot(W)  # shape (N, C)
    probs = softmax(scores)  # shape (N, C)
    data_loss = np.sum(-np.log(probs[np.arange(N), y])) / N
    reg_loss = 0.5 * reg * np.sum(W ** 2)
    loss = data_loss + reg_loss

    dscores = probs.copy()  # shape (N, C)
    dscores[np.arange(N), y] -= 1.0
    dscores /= N
    dW = np.dot(X.T, dscores) + reg * W  # shape (D, C)
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax(scores):
    scores_max = np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
