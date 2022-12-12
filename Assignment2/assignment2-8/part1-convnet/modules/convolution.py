"""
2d Convolution Module.  (c) 2021 Georgia Tech

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


class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        """
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * \
            np.random.randn(self.out_channels, self.in_channels,
                            self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        """
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        """
        out = None
        #############################################################################
        # TODO: Implement the convolution forward pass.                             #
        # Hint: 1) You may use np.pad for padding.                                  #
        #       2) You may implement the convolution with loops                     #
        #############################################################################
        # This borrows the idea from
        # https://github.com/huyouare/CS231n/blob/master/assignment2/cs231n/im2col.py

        # Get the shape of the input data
        N, C, H, W = x.shape
        fh, fw = self.kernel_size, self.kernel_size
        Hout = int(np.floor((H + 2 * self.padding - fh) / self.stride)) + 1
        Wout = int(np.floor((W + 2 * self.padding - fw) / self.stride)) + 1

        # Reshape the input data
        x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding),
                          (self.padding, self.padding)), constant_values=0.)

        i0 = np.tile(np.repeat(np.arange(fh), fw), C)
        i1 = self.stride * np.repeat(np.arange(Hout), Wout)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)

        j0 = np.tile(np.arange(fw), fh * C)
        j1 = self.stride * np.tile(np.arange(Wout), Hout)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)

        k = np.repeat(np.arange(C), fh * fw).reshape(-1, 1)

        x_cols = x_padded[:, k, i, j]
        x_cols = x_cols.transpose(1, 2, 0).reshape(fh * fw * C, -1)

        # Reshape the weights
        weights_cols = self.weight.reshape(self.out_channels, -1)
        out = np.matmul(weights_cols, x_cols) + self.bias.reshape(-1, 1)

        # Shape back the output
        out = out.reshape(self.out_channels, Hout, Wout, N)
        out = out.transpose(3, 0, 1, 2)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = (x, x_cols, Hout, Wout, k, i, j)
        return out

    def backward(self, dout):
        """
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        """
        x, x_cols, Hout, Wout, k, i, j = self.cache
        #############################################################################
        # TODO: Implement the convolution backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the convolution with loops                     #
        #       2) don't forget padding when computing dx                           #
        #############################################################################
        N, C, W, H = x.shape
        fh, fw = self.kernel_size, self.kernel_size

        # Calculate db
        # First reshape the dout
        dout_reshaped = dout.transpose(
            1, 2, 3, 0).reshape(self.out_channels, -1)
        self.db = dout_reshaped @ np.ones(dout_reshaped.shape[1])

        # Calculate the dw
        self.dw = dout_reshaped @ x_cols.T
        self.dw = self.dw.reshape(self.weight.shape)

        # Calculate the dx
        weights_cols = self.weight.reshape(self.out_channels, -1)
        dx_cols = weights_cols.T @ dout_reshaped

        # Shape back the gradient
        self.dx = np.zeros((N, C, H + 2*self.padding, W + 2*self.padding))
        dx_cols = dx_cols.reshape(C * fh * fw, -1, N)
        dx_cols = dx_cols.transpose(2, 0, 1)
        np.add.at(self.dx, (slice(None), k, i, j), dx_cols)

        if self.padding != 0:
            self.dx = self.dx[:, :, self.padding:-self.padding,
                              self.padding:-self.padding]

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
