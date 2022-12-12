"""
Linear Module.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import numpy as np


class Linear:
    """
    A linear layer with weight W and bias b. Output is computed by y = Wx + b
    """

    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.in_dim, self.out_dim)
        np.random.seed(1024)
        self.bias = np.zeros(self.out_dim)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        """
        Forward pass of linear layer
        :param x: input data, (N, d1, d2, ..., dn) where the product of d1, d2, ..., dn is equal to self.in_dim
        :return: The output computed by Wx+b. Save necessary variables in cache for backward
        """
        out = None
        #############################################################################
        # TODO: Implement the forward pass.                                         #
        #    HINT: You may want to flatten the input first                          #
        #############################################################################
        x_flattened = x.reshape(x.shape[0], -1)
        out = np.matmul(x_flattened, self.weight) + self.bias

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x
        return out

    def backward(self, dout):
        """
        Computes the backward pass of linear layer
        :param dout: Upstream gradients, (N, self.out_dim)
        :return: nothing but dx, dw, and db of self should be updated
        """
        x = self.cache
        #############################################################################
        # TODO: Implement the linear backward pass.                                 #
        #############################################################################
        x_flattened = x.reshape(x.shape[0], -1)
        bias_col = np.ones((x.shape[0], 1))
        x_bias = np.column_stack((x_flattened, bias_col))

        Wb = np.row_stack((self.weight, self.bias))

        N, _ = dout.shape
        dout_reshaped = dout.reshape(N, 1, self.out_dim)
        x_reshaped = x_bias.reshape(N, -1, 1)
        dWb = np.sum(dout_reshaped * x_reshaped, axis=0)

        self.dw = dWb[:-1, :]
        self.db = dWb[-1, :]

        Wb_t = Wb.T
        dout_t_reshaped = dout.T.reshape(self.out_dim, 1, N)
        Wb_t_reshaped = Wb_t.reshape(self.out_dim, -1, 1)
        dx_t = np.sum(dout_t_reshaped * Wb_t_reshaped, axis=0)

        self.dx = dx_t[:-1, :].T
        self.dx = self.dx.reshape(x.shape)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
