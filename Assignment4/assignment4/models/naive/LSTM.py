"""
LSTM model.  (c) 2021 Georgia Tech

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
import torch
import torch.nn as nn


class LSTM(nn.Module):
    # An implementation of naive LSTM using Pytorch Linear layers and activations
    # You will need to complete the class init function, forward function and weight initialization

    def __init__(self, input_size, hidden_size):
        """ Init function for VanillaRNN class
            Args:
                input_size (int): the number of features in the inputs.
                hidden_size (int): the size of the hidden layer
            Returns:
                None
        """
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        ################################################################################
        # TODO:                                                                        #
        #   Declare LSTM weights and attributes as you wish here.                      #
        #   You should include weights and biases regarding using nn.Parameter:        #
        #       1) i_t: input gate                                                     #
        #       2) f_t: forget gate                                                    #
        #       3) g_t: cell gate, or the tilded cell state                            #
        #       4) o_t: output gate                                                    #
        #   You also need to include correct activation functions                      #
        ################################################################################

        # i_t: input gate
        self.Wii = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.Wih = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bi = nn.Parameter(torch.Tensor(hidden_size))

        # f_t: the forget gate
        self.Wfi = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.Wfh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bf = nn.Parameter(torch.Tensor(hidden_size))

        # g_t: the cell gate
        self.Wgi = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.Wgh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bg = nn.Parameter(torch.Tensor(hidden_size))

        # o_t: the output gate
        self.Woi = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.Woh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bo = nn.Parameter(torch.Tensor(hidden_size))

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        self.init_hidden()

    def init_hidden(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x: torch.Tensor, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""

        ################################################################################
        # TODO:                                                                        #
        #   Implement the forward pass of LSTM. Please refer to the equations in the   #
        #   corresponding section of jupyter notebook. Iterate through all the time    #
        #   steps and return only the hidden and cell state, h_t and c_t.              #
        #   Note that this time you are also iterating over all of the time steps.     #
        ################################################################################
        h_t, c_t = None, None

        N, S, F = x.size()

        if init_states is None:
            h_t = torch.zeros(N, self.hidden_size)
            c_t = torch.zeros(N, self.hidden_size)
        else:
            h_t, c_t = init_states

        for i in range(S):
            x_i = x[:, i, :]

            igate_i = self.sigmoid(torch.matmul(
                x_i, self.Wii) + torch.matmul(h_t, self.Wih) + self.bi)
            fgate_i = self.sigmoid(torch.matmul(
                x_i, self.Wfi) + torch.matmul(h_t, self.Wfh) + self.bf)
            cgate_i = self.tanh(torch.matmul(
                x_i, self.Wgi) + torch.matmul(h_t, self.Wgh) + self.bg)
            ogate_i = self.sigmoid(torch.matmul(
                x_i, self.Woi) + torch.matmul(h_t, self.Woh) + self.bo)

            c_t = fgate_i * c_t + igate_i * cgate_i
            h_t = ogate_i * self.tanh(c_t)

        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        return (h_t, c_t)
