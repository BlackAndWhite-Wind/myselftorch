import numpy as np
from tensor import *


class LossFunction():
    def __init__(self):
        pass

    def Loss(self, output, label):
        pass


# The MSE class inherits from the LossFunction class and implements the Loss function.
class MSE(LossFunction):
    def __init__(self):
        self.loss_function = 'MSE'

    def Loss(self, output, label):
        return np.sum((output.data - label.data) *
                      (output.data - label.data)) / 2, (output - label)


# `Exp` is a subclass of `LossFunction` that implements the `Loss` method
class Exp(LossFunction):
    def __init__(self):
        self.loss_function = 'Exp'

    def Loss(self, output, label):
        origin = np.exp(-(output.data * label.data))
        return origin, tensor(origin * (-label.data))


# The Cross_Entropy class inherits from the LossFunction class and implements the Loss function.
class cross_entropy(LossFunction):
    def __init__(self):
        self.loss_function = 'cross_entropy'

    def Loss(self, output, label):
        origin = -np.sum(label.data * np.log(output.data) +
                         (1 - label.data) * np.log(1 - output.data))
        return origin, (output - label)


# > The class is a template for the optimizer
class Optim():
    def __init__(self):
        pass

    def zero_grad(self):
        param = self.model.parameters()
        for Tensor in param:
            if Tensor.grad is not None:
                Tensor.grad.data -= Tensor.grad.data
            else:
                print('No grad')

    def Loss(self, output, label, LossFunc):
        pass

    def Step(self):
        pass


# The SGD class inherits from the Optim class and implements the Loss function.
class SGD(Optim):
    def __init__(self, model, lr, minibatch=1):
        self.model = model
        self.lr = lr
        self.MiniBatch = minibatch
        self.dLoss = None

    def Loss(self, output, label, LossFunc):
        self.loss_fn = LossFunc.loss_function
        self.loss, self.dLoss = LossFunc.Loss(output, label)
        self.loss = self.loss / self.MiniBatch
        self.output = output

    def Step(self):
        param = self.model.parameters()
        if self.loss_fn == 'cross_entropy':
            self.output.grad_fn.x.backward(self.dLoss)
        else:
            self.output.backward(self.dLoss)
        for Tensor in param:
            Tensor.data -= self.lr * Tensor.grad.data / self.MiniBatch


# The Adam class inherits from the Optim class and implements the Loss function.
class Hebb(Optim):
    def __init__(self, t, p, w=None):
        self.t = t
        self.p = p
        self.w = w

    def Step(self):
        if not isinstance(self.w, type(None)):
            self.w += np.dot(self.t, self.p.T)
        else:
            self.w = np.dot(self.t, self.p.T)


# perceptron
class Perceptron(Optim):
    def __init__(self, ponits, targets, w, b):
        self.points = ponits
        self.targets = targets
        self.w = w
        self.b = b

    def Step(self):
        for i in range(self.points.shape[1]):
            temp = np.dot(self.w, self.points[:, i]) + self.b
            temp[temp < 0] = 0
            temp[temp > 0] = 1
            e = self.targets[:, i] - temp
            e.resize(e.shape[0], 1)
            self.w += np.dot(
                e, np.resize(self.points[:, i].T, [1, self.points.shape[0]]))
            self.b += e

    def get_w(self):
        return self.w

    def get_b(self):
        return self.b
