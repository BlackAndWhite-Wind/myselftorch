from tensor import *
import numpy as np
from all_function import *


class Functional():
    def __init__(self) -> None:
        pass

    def forward(self):
        pass

    def backward(self):
        pass


class addm_backward(Functional):
    def __init__(self) -> None:
        super(addm_backward, self).__init__()

    def forward(self, x, w, b):
        self.save_backward(x, w, b)
        return multiple_matrix(w, x) + b

    def backward(self, grad_output=tensor([1])):
        x, w, b = self.x, self.w, self.b
        grad_w = multiple_matrix(grad_output, x.trans())
        grad_x = multiple_matrix(w.trans(), grad_output)
        grad_b = grad_output.data.sum(axis=1)
        grad_b.resize([grad_b.shape[0], 1])
        grad_b = tensor(grad_b)
        if x.requires_grad == True:
            if x.grad is None:
                x.grad = grad_x
            else:
                x.grad += grad_x
            if x.is_leaf == False:
                x.grad_fn.backward(grad_output=x.grad)
        if w.requires_grad == True:
            if w.grad is None:
                w.grad = grad_w
            else:
                w.grad += grad_w
        if b.requires_grad == True:
            if b.grad is None:
                b.grad = grad_b
            else:
                b.grad += grad_b
        return grad_x, grad_w, grad_b

    def __call__(self, x, w, b):
        return self.forward(x, w, b)

    def save_backward(self, x, w, b):
        self.x = x
        self.w = w
        self.b = b


class relu_backward(Functional):
    def __init__(self) -> None:
        super(relu_backward, self).__init__()

    def forward(self, x, inplace):
        self.save_backward(x)
        return relu(x, inplace)

    def __call__(self, x, inplace):
        return self.forward(x, inplace)

    def backward(self, grad_output=tensor([1])):
        x = self.x

        grad_x = drelu(x) * grad_output

        if x.requires_grad == True:
            if x.grad is None:
                x.grad = grad_x

            else:
                x.grad += grad_x
            if x.is_leaf == False:
                x.grad_fn.backward(grad_output=x.grad)

    def save_backward(self, x):
        self.x = x


class sigmoid_backward(Functional):
    def __init__(self) -> None:
        super(sigmoid_backward, self).__init__()

    def forward(self, x):
        self.save_backward(x)
        return sigmoid(x)

    def __call__(self, x):
        return self.forward(x)

    def backward(self, grad_output=tensor([1])):

        x = self.x
        grad_x = dsigmoid(x) * grad_output

        if x.requires_grad == True:
            if x.grad is None:
                x.grad = grad_x
            else:
                x.grad += grad_x
            if x.is_leaf == False:
                x.grad_fn.backward(grad_output=x.grad)

    def save_backward(self, x):
        self.x = x


class softmax_backward(Functional):
    def __init__(self) -> None:
        super(softmax_backward, self).__init__()

    def forward(self, x):
        self.save_backward(x)
        output = softmax(x.data.copy())
        return tensor(output, is_leaf=False, requires_grad=True)

    def __call__(self, x):
        return self.forward(x)

    def save_backward(self, x):
        self.x = x


class conv2d_backward(Functional):
    def __init__(self) -> None:
        super(conv2d_backward, self).__init__()

    def __call__(self, x, w):
        return self.forward(x, w)

    def forward(self, x, w):
        self.save_backward(x, w)

        self.temp_x, self.temp_w = get_rows(x.data, w.data)

        y = anti_cols(np.dot(self.temp_x, self.temp_w), x.shape[1], x.shape[2],
                      w.shape[2], w.shape[3])
        return tensor(y, is_leaf=False, requires_grad=True)

    def backward(self, grad_output=tensor([1])):
        grad_temp = cols(grad_output.data)
        ###############
        temp_x_grad = np.dot(grad_temp, self.temp_w.T)
        ###############
        temp_w_grad = np.dot(self.temp_x.T, grad_temp)
        x_grad = tensor(build_a(temp_x_grad, self.x.data, self.w.data))
        w_grad = tensor(
            build_w(temp_w_grad, self.w.shape[1], self.w.shape[2],
                    self.w.shape[3]))

        if self.x.requires_grad == True:
            if self.x.grad is None:
                self.x.grad = x_grad
            else:
                self.x.grad += x_grad
            if self.x.is_leaf == False:
                self.x.grad_fn.backward(grad_output=self.x.grad)
        if self.w.requires_grad == True:
            if self.w.grad is None:
                self.w.grad = w_grad
            else:
                self.w.grad += w_grad

    def save_backward(self, x, w):
        self.x = x
        self.w = w


class average_backward(Functional):
    def __init__(self) -> None:
        super(average_backward, self).__init__()

    def __call__(self, x, kernel_size):
        return self.forward(x, kernel_size)

    def forward(self, x, kernel_size):
        self.size = kernel_size
        self.x = x

        output = Average(x.data, kernel_size)
        return tensor(output, is_leaf=False, requires_grad=True)

    def backward(self, grad_output=tensor([1])):
        temp_grad = grad_output.data
        x_grad = tensor(dAverage(temp_grad, self.size, self.x.data))

        if self.x.requires_grad == True:
            if self.x.grad is None:
                self.x.grad = x_grad
            else:
                self.x.grad += x_grad
            if self.x.is_leaf == False:
                self.x.grad_fn.backward(grad_output=self.x.grad)


class max_backward(Functional):
    def __init__(self) -> None:
        super(max_backward, self).__init__()

    def __call__(self, x, kernel_size):
        return self.forward(x, kernel_size)

    def forward(self, x, kernel_size):
        self.size = kernel_size
        self.x = x

        output, self.record = Maxpool(x.data, kernel_size)
        return tensor(output, is_leaf=False, requires_grad=True)

    def backward(self, grad_output=tensor([1])):
        temp_grad = grad_output.data
        x_grad = tensor(dMaxpool(temp_grad, self.size, self.record))

        if self.x.requires_grad == True:
            if self.x.grad is None:
                self.x.grad = x_grad
            else:
                self.x.grad += x_grad
            if self.x.is_leaf == False:
                self.x.grad_fn.backward(grad_output=self.x.grad)


class DimsBackward(Functional):
    def __init__(self) -> None:
        super(DimsBackward, self).__init__()

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.x = x

        output = np.resize(x.data, [x.shape[0] * x.data[0].size, 1])

        return tensor(output, is_leaf=False, requires_grad=True)

    def backward(self, grad_output):
        temp = grad_output.data
        x_grad = tensor(
            np.resize(temp,
                      [self.x.shape[0], self.x.shape[1], self.x.shape[2]]))
        if self.x.requires_grad == True:
            if self.x.grad is None:
                self.x.grad = x_grad
            else:
                self.x.grad += x_grad
            if self.x.is_leaf == False:
                self.x.grad_fn.backward(grad_output=self.x.grad)
