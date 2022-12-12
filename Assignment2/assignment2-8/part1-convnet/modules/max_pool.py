"""
2d Max Pooling Module.  (c) 2021 Georgia Tech

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


class MaxPooling:
    """
    Max Pooling of input
    """

    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None

    def forward(self, x):
        """
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        """
        out = None
        #############################################################################
        # TODO: Implement the max pooling forward pass.                             #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #############################################################################
        # To implement without loop, needs some tricks
        # Reshape the matrix to be (N, C, Hout, Wout, kernel_size, kernel_size)
        # This borrows the idea from
        # https://github.com/huyouare/CS231n/blob/master/assignment2/cs231n/im2col.py
        N, C, H, W = x.shape
        x_r = x.reshape(N * C, 1, H, W)
        C = 1
        fh = self.kernel_size
        fw = self.kernel_size
        H_out = int(np.floor((H - fh) / self.stride) + 1)
        W_out = int(np.floor((W - fw) / self.stride) + 1)

        # calculate the corresponding index
        i0 = np.tile(np.repeat(np.arange(fh), fw), C)
        i1 = self.stride * np.repeat(np.arange(H_out), W_out)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)

        j0 = np.tile(np.arange(fw), fh * C)
        j1 = self.stride * np.tile(np.arange(W_out), H_out)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)

        k = np.repeat(np.arange(C), fh * fw).reshape(-1, 1)

        # get the reshaped matrix
        xcol = x_r[:, k, i, j]
        xcol = xcol.transpose(1, 2, 0).reshape(fh * fw * C, -1)

        out = np.max(xcol, axis=0)
        C = x.shape[1]
        out = out.reshape(H_out, W_out, N, C)
        out = out.transpose(2, 3, 0, 1)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = (x, H_out, W_out)
        return out

    def backward(self, dout):
        """
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return:
        """
        x, H_out, W_out = self.cache
        #############################################################################
        # TODO: Implement the max pooling backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #       2) You may find np.unravel_index useful                             #
        #############################################################################
        N, C, H, W = x.shape
        x_r = x.reshape(N * C, 1, H, W)
        C = 1
        fh = self.kernel_size
        fw = self.kernel_size
        H_out = int(np.floor((H - fh) / self.stride) + 1)
        W_out = int(np.floor((W - fw) / self.stride) + 1)

        # calculate the corresponding index
        i0 = np.tile(np.repeat(np.arange(fh), fw), C)
        i1 = self.stride * np.repeat(np.arange(H_out), W_out)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)

        j0 = np.tile(np.arange(fw), fh * C)
        j1 = self.stride * np.tile(np.arange(W_out), H_out)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)

        k = np.repeat(np.arange(C), fh * fw).reshape(-1, 1)

        # get the reshaped matrix
        xcol = x_r[:, k, i, j]
        xcol = xcol.transpose(1, 2, 0).reshape(fh * fw * C, -1)

        # get the max index
        maxindices = np.argmax(xcol, axis=0)

        # create a dx matrix matching xcol
        dxcol = np.zeros_like(xcol)
        # flatten the dout matrix
        dout_raveled = dout.transpose(2, 3, 0, 1).ravel()
        # assign the corresponding gradient
        dxcol[maxindices, np.arange(xcol.shape[1])] = dout_raveled

        # transform back dx
        N, C, H, W = x.shape
        dxcol_reshaped = dxcol.reshape(fh * fw, -1, N * C)
        dxcol_reshaped = dxcol_reshaped.transpose(2, 0, 1)
        self.dx = np.zeros((N * C, 1, H, W))
        np.add.at(self.dx, (slice(None), k, i, j), dxcol_reshaped)
        self.dx = self.dx.reshape(N, C, H, W)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
