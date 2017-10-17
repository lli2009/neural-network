import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import utils

"""
    Definitions:
        W: weights, dimension: j x i where j is the number of nodes in
           next layer and i is the number of nodes in current layer
        b: bias, dimension: 1 x j where j is the number of nodes in next layer
        X: data, dimension: m x i where m is the number of examples and i is
           the number of nodes in current layer
        y: output, dimension: m x 1 where m is the number of examples, the
           value of y is between 0 to k, where k is the number of output
           classes

    Basic operations:
        linear operation: a = np.dot(X, np.transpose(W)) + b

"""


class AddOperation(object):
    @staticmethod
    def forward(mul, b):
        return np.add(mul, b)

    @staticmethod
    def backward(mul, dZ):
        dmul = dZ * np.ones_like(mul)
        db = np.dot(np.ones((1, dZ.shape[0])), dZ)
        return dmul, db


class MultiplyOperation(object):
    @staticmethod
    def forward(W, X):
        return np.dot(X, np.transpose(W))

    @staticmethod
    def backward(W, X, dZ):
        dW = np.dot(np.transpose(dZ), X)
        dX = np.dot(dZ, W)
        return dW, dX


class SigmoidActivation(object):
    @staticmethod
    def forward(add):
        return 1 / (np.exp(-add) + 1)

    @staticmethod
    def backward(add, dZ):
        da = dZ * add * (1 - add)
        return da


class SoftMax(object):
    @staticmethod
    def predict(a):
        normalized_a = SoftMax._normalize(a)
        return np.exp(normalized_a) / np.sum(np.exp(normalized_a), axis=1,
                                             keepdims=True)

    @staticmethod
    def loss(a, y):
        num_examples = a.shape[0]
        probs = SoftMax.predict(a)
        x_entropy = np.sum(
            -np.log(probs[range(num_examples), y])) / num_examples
        return x_entropy

    @staticmethod
    def accuracy(a, y):
        probs = SoftMax.predict(a)
        predict_values = np.argmax(probs, axis=1)
        assert_ret = (predict_values == y)
        correct = assert_ret[assert_ret]
        return len(correct) * 1.0 / len(assert_ret)

    @staticmethod
    def diff(a, y):
        num_examples = a.shape[0]
        predict_value = SoftMax.predict(a)
        predict_value[range(num_examples), y] -= 1
        return predict_value

    @staticmethod
    def _normalize(a):
        m = np.max(a, axis=1, keepdims=True)
        normalized_a = a - m
        return normalized_a


class Sequential(object):
    def __init__(self, layer_dim):
        self.W = []
        self.b = []
        for layer in range(len(layer_dim) - 1):
            s_current = layer_dim[layer]
            s_next = layer_dim[layer + 1]
            self.W.append(np.random.randn(s_next, s_current))
            self.b.append(np.random.randn(s_next).reshape(1, s_next))

    def train(self, X, y, epoches, learning_rate=0.01, print_metrics=False):

        last_forward = None
        for epoch in range(epoches):
            input_values = X
            forward = []
            for layer in range(len(self.W)):
                mul = MultiplyOperation.forward(self.W[layer], input_values)
                add = AddOperation.forward(mul, self.b[layer])
                a = SigmoidActivation.forward(add)
                forward.append((np.array(input_values), mul, add, a))
                input_values = a

            last_a = forward[-1][3]
            dZ = SoftMax.diff(last_a, y)
            for layer in range(len(self.W) - 1, -1, -1):
                input_values, mul, add, a = forward[layer]
                da = SigmoidActivation.backward(a, dZ)
                dmul, db = AddOperation.backward(mul, da)
                dW, dX = MultiplyOperation.backward(self.W[layer],
                                                    input_values, dmul)
                dZ = dX
                self.W[layer] -= learning_rate * dW
                self.b[layer] -= learning_rate * db

            if print_metrics:
                accuracy = SoftMax.accuracy(forward[-1][-1], y)
                loss = SoftMax.loss(forward[-1][3], y)
                print "Epoch: {} - Accuracy: {}, Loss: {}".format(
                    epoch, accuracy, loss)

    def predict(self, X):
        input_values = X
        for layer in range(len(self.W)):
            mul = MultiplyOperation.forward(self.W[layer], input_values)
            add = AddOperation.forward(mul, self.b[layer])
            input_values = SigmoidActivation.forward(add)
        probs = SoftMax.predict(input_values)
        return np.argmax(probs, axis=1)


def run():
    # Generate a dataset and plot it
    np.random.seed(0)
    X, y = sklearn.datasets.make_moons(200, noise=0.20)
    utils.plot_training_examples(X, y)

    layers_dim = [2, 3, 2]

    model = Sequential(layers_dim)
    model.train(X, y, epoches=20000, learning_rate=0.01, print_metrics=True)

    utils.plot_decision_boundary(lambda x: model.predict(x), X, y)
    utils.plot_show()


if __name__ == '__main__':
    run()
