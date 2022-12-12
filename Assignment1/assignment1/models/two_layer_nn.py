# Do not use packages that are not in standard distribution of python
from ._base_network import _baseNetwork
import numpy as np
np.random.seed(1024)


class TwoLayerNet(_baseNetwork):
    def __init__(self, input_size=28 * 28, num_classes=10, hidden_size=128):
        super().__init__(input_size, num_classes)

        self.hidden_size = hidden_size
        self._weight_init()

    def _weight_init(self):
        '''
        initialize weights of the network
        :return: None; self.weights is filled based on method
        - W1: The weight matrix of the first layer of shape (num_features, hidden_size)
        - b1: The bias term of the first layer of shape (hidden_size,)
        - W2: The weight matrix of the second layer of shape (hidden_size, num_classes)
        - b2: The bias term of the second layer of shape (num_classes,)
        '''

        # initialize weights
        self.weights['b1'] = np.zeros(self.hidden_size)
        self.weights['b2'] = np.zeros(self.num_classes)
        np.random.seed(1024)
        self.weights['W1'] = 0.001 * \
            np.random.randn(self.input_size, self.hidden_size)
        np.random.seed(1024)
        self.weights['W2'] = 0.001 * \
            np.random.randn(self.hidden_size, self.num_classes)

        # initialize gradients to zeros
        self.gradients['W1'] = np.zeros((self.input_size, self.hidden_size))
        self.gradients['b1'] = np.zeros(self.hidden_size)
        self.gradients['W2'] = np.zeros((self.hidden_size, self.num_classes))
        self.gradients['b2'] = np.zeros(self.num_classes)

    def forward(self, X, y, mode='train'):
        '''
        The forward pass of the two-layer net. The activation function used in between the two layers is sigmoid, which
        is to be implemented in self.,sigmoid.
        The method forward should compute the loss of input batch X and gradients of each weights.
        Further, it should also compute the accuracy of given batch. The loss and
        accuracy are returned by the method and gradients are stored in self.gradients

        :param X: a batch of images (N, input_size)
        :param y: labels of images in the batch (N,)
        :param mode: if mode is training, compute and update gradients;else, just return the loss and accuracy
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
            self.gradients: gradients are not explicitly returned but rather updated in the class member self.gradients
        '''
        loss = None
        accuracy = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the forward process:                                      #
        #        1) Call sigmoid function between the two layers for non-linearity  #
        #        2) The output of the second layer should be passed to softmax      #
        #        function before computing the cross entropy loss                   #
        #    2) Compute Cross-Entropy Loss and batch accuracy based on network      #
        #       outputs                                                             #
        #############################################################################
        batch_size = X.shape[0]
        # W1_bar: (input_size + 1, hidden_size)
        W1_bar = np.concatenate(
            (self.weights['W1'], self.weights['b1'].reshape(1, self.hidden_size)), axis=0)
        # X_bar: (N, input_size + 1)
        X_bar = np.concatenate((X, np.ones((batch_size, 1))), axis=1)
        # output1: (N, hidden_size)
        output1 = np.matmul(X_bar, W1_bar)

        # sig_output: (N, hidden_size)
        sig_output = self.sigmoid(output1)

        # W2_bar: (hidden_size + 1, num_classes)
        W2_bar = np.concatenate(
            (self.weights['W2'], self.weights['b2'].reshape(1, self.num_classes)), axis=0)
        # sig_bar: (N, hidden_size + 1)
        sig_bar = np.concatenate(
            (sig_output, np.ones((batch_size, 1))), axis=1)
        # output2: (N, num_classes)
        output2 = np.matmul(sig_bar, W2_bar)

        # x_pred: (N, num_classes)
        x_pred = self.softmax(output2)
        loss = self.cross_entropy_loss(x_pred, y)
        accuracy = self.compute_accuracy(x_pred, y)

        if mode != 'train':
            return loss, accuracy

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the backward process:                                     #
        #        1) Compute gradients of each weight and bias by chain rule         #
        #        2) Store the gradients in self.gradients                           #
        #    HINT: You will need to compute gradients backwards, i.e, compute       #
        #          gradients of W2 and b2 first, then compute it for W1 and b1      #
        #          You may also want to implement the analytical derivative of      #
        #          the sigmoid function in self.sigmoid_dev first                   #
        #############################################################################
        gradient2 = None
        gradient2 = x_pred
        gradient2[range(batch_size), y] -= 1.
        tmpX = sig_bar.reshape(*sig_bar.shape, 1).repeat(self.num_classes, 2)
        tmpG = gradient2.reshape(gradient2.shape[0], 1, gradient2.shape[1])
        tmpG = tmpG.repeat(self.hidden_size + 1, 1)
        tmpfinalG2 = tmpX * tmpG
        tmpfinalG2 = np.sum(tmpfinalG2, axis=0) / batch_size
        self.gradients['W2'] = tmpfinalG2[:-1]
        self.gradients['b2'] = tmpfinalG2[-1]

        gradient1 = np.matmul(gradient2, self.weights['W2'].T)
        gradient1 = gradient1 * self.sigmoid_dev(output1)
        tmpX = X_bar.reshape(*X_bar.shape, 1).repeat(self.hidden_size, 2)
        tmpG = gradient1.reshape(gradient1.shape[0], 1, gradient1.shape[1])
        tmpG = tmpG.repeat(self.input_size + 1, 1)
        tmpfinalG1 = tmpX * tmpG
        tmpfinalG1 = np.sum(tmpfinalG1, axis=0) / batch_size
        self.gradients['W1'] = tmpfinalG1[:-1]
        self.gradients['b1'] = tmpfinalG1[-1]

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return loss, accuracy
