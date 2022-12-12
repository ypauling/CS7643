# Do not use packages that are not in standard distribution of python
import numpy as np

from ._base_network import _baseNetwork


class SoftmaxRegression(_baseNetwork):
    def __init__(self, input_size=28*28, num_classes=10):
        '''
        A single layer softmax regression. The network is composed by:
        a linear layer without bias => (optional ReLU activation) => Softmax
        :param input_size: the input dimension
        :param num_classes: the number of classes in total
        '''
        super().__init__(input_size, num_classes)
        self._weight_init()

    def _weight_init(self):
        '''
        initialize weights of the single layer regression network. No bias term included.
        :return: None; self.weights is filled based on method
        - W1: The weight matrix of the linear layer of shape (num_features, hidden_size)
        '''
        np.random.seed(1024)
        self.weights['W1'] = 0.001 * \
            np.random.randn(self.input_size, self.num_classes)
        self.gradients['W1'] = np.zeros((self.input_size, self.num_classes))

    def forward(self, X, y, mode='train'):
        '''
        Compute loss and gradients using softmax with vectorization.

        :param X: a batch of image (N, 28x28)
        :param y: labels of images in the batch (N,)
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
        '''
        loss = None
        gradient = None
        accuracy = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the forward process and compute the Cross-Entropy loss    #
        #    2) Compute the gradient of the loss with respect to the weights        #
        # Hint:                                                                     #
        #   Store your intermediate outputs before ReLU for backwards               #
        #############################################################################
        linear_output = np.matmul(X, self.weights['W1'])
        relu_output = self.ReLU(linear_output)
        softmax_output = self.softmax(relu_output)
        loss = self.cross_entropy_loss(softmax_output, y)
        accuracy = self.compute_accuracy(softmax_output, y)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        if mode != 'train':
            return loss, accuracy

        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the backward process:                                     #
        #        1) Compute gradients of each weight by chain rule                  #
        #        2) Store the gradients in self.gradients                           #
        #############################################################################
        batch_size = y.shape[0]
        gradient = softmax_output
        gradient[range(batch_size), y] -= 1.
        # gradient = gradient / batch_size

        gradient = gradient * self.ReLU_dev(linear_output)
        # expand the X matrix along z-axis -> (N, 28x28, num_classes)
        X_reshape = X.reshape(*X.shape, 1).repeat(self.num_classes, 2)
        # expand the gradient marix along y-axis -> (N, 28x28, num_classes)
        g_reshape = gradient.reshape(gradient.shape[0], 1, gradient.shape[1])
        g_reshape = g_reshape.repeat(self.input_size, 1)
        # Multiply the two reshaped matrices, size matched
        final_g = X_reshape * g_reshape
        # average over batch
        self.gradients['W1'] = np.sum(final_g, axis=0) / batch_size

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss, accuracy
