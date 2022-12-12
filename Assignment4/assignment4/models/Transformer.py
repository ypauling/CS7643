"""
Transformer model.  (c) 2021 Georgia Tech

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
from torch import nn
import random

# Do not modify these imports.


class TransformerTranslator(nn.Module):
    """
    A single-layer Transformer which encodes a sequence of text and
    performs binary classification.

    The model has a vocab size of V, works on
    sequences of length T, has an hidden dimension of H, uses word vectors
    also of dimension H, and operates on minibatches of size N.
    """

    def __init__(self, input_size, output_size, device, hidden_dim=128, num_heads=2,
                 dim_feedforward=2048, dim_k=96, dim_v=96, dim_q=96, max_length=43):
        """
        :param input_size: the size of the input, which equals to the number of words in source language vocabulary
        :param output_size: the size of the output, which equals to the number of words in target language vocabulary
        :param hidden_dim: the dimensionality of the output embeddings that go into the final layer
        :param num_heads: the number of Transformer heads to use
        :param dim_feedforward: the dimension of the feedforward network model
        :param dim_k: the dimensionality of the key vectors
        :param dim_q: the dimensionality of the query vectors
        :param dim_v: the dimensionality of the value vectors
        """
        super(TransformerTranslator, self).__init__()
        assert hidden_dim % num_heads == 0

        self.num_heads = num_heads
        self.word_embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_q = dim_q

        seed_torch(0)

        ##############################################################################
        # TODO:
        # Deliverable 1: Initialize what you need for the embedding lookup.          #
        # You will need to use the max_length parameter above.                       #
        # This should take 1-2 lines.                                                #
        # Initialize the word embeddings before the positional encodings.            #
        # Don’t worry about sine/cosine encodings- use positional encodings.         #
        ##############################################################################
        self.word_embed = nn.Embedding(input_size, hidden_dim)
        self.pos_embed = nn.Embedding(max_length, hidden_dim)

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        ##############################################################################
        # Deliverable 2: Initializations for multi-head self-attention.              #
        # You don't need to do anything here. Do not modify this code.               #
        ##############################################################################

        # Head #1
        self.k1 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v1 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q1 = nn.Linear(self.hidden_dim, self.dim_q)

        # Head #2
        self.k2 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v2 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q2 = nn.Linear(self.hidden_dim, self.dim_q)

        self.softmax = nn.Softmax(dim=2)
        self.attention_head_projection = nn.Linear(
            self.dim_v * self.num_heads, self.hidden_dim)
        self.norm_mh = nn.LayerNorm(self.hidden_dim)

        ##############################################################################
        # TODO:
        # Deliverable 3: Initialize what you need for the feed-forward layer.        #
        # Don't forget the layer normalization.                                      #
        ##############################################################################

        self.ffl_linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.ffl_linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.ffl_relu = nn.ReLU()
        self.ffl_norm = nn.LayerNorm(self.hidden_dim)

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        ##############################################################################
        # TODO:
        # Deliverable 4: Initialize what you need for the final layer (1-2 lines).   #
        ##############################################################################

        self.final_linear = nn.Linear(hidden_dim, output_size)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

    def forward(self, inputs):
        """
        This function computes the full Transformer forward pass.
        Put together all of the layers you've developed in the correct order.

        :param inputs: a PyTorch tensor of shape (N,T). These are integer lookups.

        :returns: the model outputs. Should be normalized scores of shape (N,1).
        """

        #############################################################################
        # TODO:
        # Deliverable 5: Implement the full Transformer stack for the forward pass. #
        # You will need to use all of the methods you have previously defined above.#
        # You should only be calling ClassificationTransformer class methods here.  #
        #############################################################################
        outputs = None

        embed = self.embed(inputs)
        mh_att = self.multi_head_attention(embed)
        ffl_output = self.feedforward_layer(mh_att)
        outputs = self.final_layer(ffl_output)

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs

    def embed(self, inputs):
        """
        :param inputs: intTensor of shape (N,T)
        :returns embeddings: floatTensor of shape (N,T,H)
        """
        embeddings = None
        #############################################################################
        # TODO:
        # Deliverable 1: Implement the embedding lookup.                            #
        # Note: word_to_ix has keys from 0 to self.vocab_size - 1                   #
        # This will take a few lines.                                               #
        #############################################################################
        word_embeddings = self.word_embed(inputs)

        N, SL = inputs.size()
        pos_matrix = torch.tensor(
            range(SL), dtype=torch.long, device=inputs.device)
        pos_embeddings = self.pos_embed(pos_matrix)

        embeddings = word_embeddings + pos_embeddings

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return embeddings

    def multi_head_attention(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)

        Traditionally we'd include a padding mask here, so that pads are ignored.
        This is a simplified implementation.
        """

        #############################################################################
        # TODO:
        # Deliverable 2: Implement multi-head self-attention followed by add + norm.#
        # Use the provided 'Deliverable 2' layers initialized in the constructor.   #
        #############################################################################
        outputs = None

        K1 = self.k1(inputs)
        V1 = self.v1(inputs)
        Q1 = self.q1(inputs)

        scores1 = torch.matmul(Q1, K1.transpose(-2, -1)) / np.sqrt(self.dim_k)
        scores1 = self.softmax(scores1)
        output1 = torch.matmul(scores1, V1)

        K2 = self.k2(inputs)
        V2 = self.v2(inputs)
        Q2 = self.q2(inputs)

        scores2 = torch.matmul(Q2, K2.transpose(-2, -1)) / np.sqrt(self.dim_k)
        scores2 = self.softmax(scores2)
        output2 = torch.matmul(scores2, V2)

        sublayer_out = self.attention_head_projection(
            torch.cat((output1, output2), dim=2))
        outputs = self.norm_mh(inputs + sublayer_out)

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs

    def feedforward_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        """

        #############################################################################
        # TODO:
        # Deliverable 3: Implement the feedforward layer followed by add + norm.    #
        # Use a ReLU activation and apply the linear layers in the order you        #
        # initialized them.                                                         #
        # This should not take more than 3-5 lines of code.                         #
        #############################################################################
        outputs = None

        outputs = self.ffl_linear1(inputs)
        outputs = self.ffl_relu(outputs)
        outputs = self.ffl_linear2(outputs)
        outputs = self.ffl_norm(outputs + inputs)

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs

    def final_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,V)
        """

        #############################################################################
        # TODO:
        # Deliverable 4: Implement the final layer for the Transformer Translator.  #
        # This should only take about 1 line of code.                               #
        #############################################################################
        outputs = None

        outputs = self.final_linear(inputs)

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
