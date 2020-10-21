# Maldikar, Akshaj Prakash
# 1001-679-565
# 2020-02-16
# Assignment-01-01

import numpy as np


class Perceptron(object):
    def __init__(self, input_dimensions=2,number_of_nodes=4):
        """
        Initialize Perceptron model and set all the weights and biases to random numbers.
        :param input_dimensions: The number of dimensions of the input data
        :param number_of_nodes: Note that number of neurons in the model is equal to the number of classes.
        """
        self.weights = None
        self.input_dimensions = input_dimensions
        self.number_of_nodes = number_of_nodes
        self.initialize_weights()
        
    def initialize_weights(self,seed=None):
        """
        Initialize the weights, initalize using random numbers.
        If seed is given, then this function should
        use the seed to initialize the weights to random numbers.
        :param seed: Random number generator seed.
        :return: None
        """
        if seed != None:
            np.random.seed(seed)
        self.weights = np.random.randn(self.number_of_nodes, self.input_dimensions + 1)

    def set_weights(self, W):
        """
        This function sets the weight matrix (Bias is included in the weight matrix).
        :param W: weight matrix
        :return: None if the input matrix, w, has the correct shape.
        If the weight matrix does not have the correct shape, this function
        should not change the weight matrix and it should return -1.
        """
        if self.weights.shape != W.shape:
            return -1
        self.weights = W

    def get_weights(self):
        """
        This function should return the weight matrix(Bias is included in the weight matrix).
        :return: Weight matrix
        """
        return self.weights

    def predict(self, X):
        """
        Make a prediction on a batach of inputs.
        :param X: Array of input [input_dimensions,n_samples]
        :return: Array of model [number_of_nodes ,n_samples]
        Note that the activation function of all the nodes is hard limit.
        """
        X_dash = self.add_bias_row_to_input(X)
        net = np.matmul(self.weights, X_dash)
        A = np.where(net > 0, 1, 0)
        return A

    def train(self, X, Y, num_epochs=10,  alpha=0.1):
        """
        Given a batch of input and desired outputs, and the necessary hyperparameters (num_epochs and alpha),
        this function adjusts the weights using Perceptron learning rule.
        Training should be repeated num_epochs times.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes ,n_samples]
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :return: None
        """

        number_of_samples = X.shape[1]
        X_dash = self.add_bias_row_to_input(X)
        for _ in range(num_epochs):
            A = self.predict(X)
            for sample in range(number_of_samples):
                e = np.subtract(Y[:, sample: sample+1], A[:, sample: sample+1])
                X_for_a_sample = X_dash[:, sample: sample+1]
                X_transpose = X_for_a_sample.transpose()
                self.weights = self.weights + np.matmul(np.multiply(alpha, e), X_transpose)

    def calculate_percent_error(self,X, Y):
        """
        Given a batch of input and desired outputs, this function calculates percent error.
        For each input sample, if the output is not the same as the desired output, Y,
        then it is considered one error. Percent error is 100*(number_of_errors/ number_of_samples).
        :param X: Array of input [input `_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes ,n_samples]
        :return percent_error
        """
        A = self.predict(X)
        number_of_samples = X.shape[1]
        error_count = 0
        for sample in range(number_of_samples):
            if not np.array_equal(Y[:, sample: sample + 1], A[:, sample: sample + 1]):
                error_count += 1

        return 100 * (error_count / number_of_samples)

    def add_bias_row_to_input(self, X):
        """
        This function appends the bias row to to the input matrix.
        :param X: Input matrix
        :return: Input matrix with bias row added
        """
        bias_input_row = self.create_bias_input_row(X)
        new_X_with_bias = np.append(bias_input_row, X, axis=0)  # Append bias row to input matrix
        return new_X_with_bias

    @staticmethod
    def create_bias_input_row(X):
        """
        This function returns a numpy row of 1's  needed for bias.
        :param X: Input Matrix
        :return: numpy row of 1's eg: [[1, 1, 1, ...]]
        """
        n_samples = X.shape[1]
        bias_row = np.ones((1, n_samples))
        return bias_row


if __name__ == "__main__":
    input_dimensions = 2
    number_of_nodes = 2

    model = Perceptron(input_dimensions=input_dimensions, number_of_nodes=number_of_nodes)
    model.initialize_weights(seed=2)
    X_train = np.array([[-1.43815556, 0.10089809, -1.25432937, 1.48410426],
                        [-1.81784194, 0.42935033, -1.2806198, 0.06527391]])
    print(model.predict(X_train))
    Y_train = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
    print("****** Model weights ******\n",model.get_weights())
    print("****** Input samples ******\n",X_train)
    print("****** Desired Output ******\n",Y_train)
    percent_error=[]
    for k in range (20):
        model.train(X_train, Y_train, num_epochs=1, alpha=0.1)
        percent_error.append(model.calculate_percent_error(X_train,Y_train))
    print("******  Percent Error ******\n",percent_error)
    print("****** Model weights ******\n",model.get_weights())
