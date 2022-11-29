from typing import OrderedDict
import numpy as np
from tensor import tensor
import auto_grad as auto
import pickle

ZEROS = 0
ONES = 1
RANDOM = 2


class Module():
    def __init__(self):
        self._parameters = OrderedDict()
        self._buffers = OrderedDict()
        self._backward_hooks = OrderedDict()
        self._forward_hooks = OrderedDict()
        self._forward_pre_hooks = OrderedDict()
        self._modules = OrderedDict()
        self.training = True

    def forward(self, x):
        pass

    def __call__(self, x):
        return self.forward(x)

    def get_children(self):
        # return all sub models of the model
        ls = OrderedDict()
        for i in self.__dict__['_modules']:
            ls[i] = self.__dict__['_modules'][i]
            ls.update(ls[i].children())
        return ls

    def get_parameters(self):
        # return the iterator of all model parameters under the model
        dic = self.get_children()
        for i in dic:
            for j in dic[i].__dict__['_parameters']:
                yield dic[i].__dict__['_parameters'][j]

    def state_dict(self):
        dic = {}
        ls = self.get_children()
        for i in ls:
            for j in ls[i].__dict__['_parameters']:
                dic[f"{i}.{j}"] = ls[i].__dict__['_parameters'][j]
        return dic

    def __setattr__(self, name, value):
        def remove_from(*dicts):
            for d in dicts:
                if name in d:
                    del d[name]

        params = self.__dict__.get('_parameters')
        #change
        if isinstance(value, tensor):
            if params is None:
                raise AttributeError(
                    "No assign parameters before Module.__init__() call")
            remove_from(self.__dict__, self._buffers, self._modules)
            self.register_parameter(name, value)
        elif params is not None and name in params:
            if value is not None:
                raise TypeError("No assign")
            self.register_parameter(name, value)

        else:
            modules = self.__dict__.get('_modules')
            if isinstance(value, Module):
                if modules is None:
                    raise AttributeError(
                        "No assign module before Module.__init__() call")
                remove_from(self.__dict__, self._parameters, self._buffers)
                modules[name] = value
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError("cannot assign as childmodule")
                modules[name] = value
            else:
                object.__setattr__(self, name, value)

    def register_parameter(self, name, param):
        if '_parameters' not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call")
        self._parameters[name] = None if param is None else param

    def __getattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'")


class Linear(Module):
    def __init__(self, input, output, type=ZEROS):
        super(Linear, self).__init__()
        if type == ZEROS:
            self.w = tensor(np.zeros([output, input]),
                            requires_grad=True,
                            is_leaf=True)
        elif type == ONES:
            self.w = tensor(np.ones([output, input]),
                            requires_grad=True,
                            is_leaf=True)
        else:
            self.w = tensor(np.random.uniform(
                -np.sqrt(6) / np.sqrt(input + output),
                np.sqrt(6) / np.sqrt(output + input),
                size=(output, input)),
                            requires_grad=True,
                            is_leaf=True)

        self.b = tensor(np.zeros([output, 1]),
                        requires_grad=True,
                        is_leaf=True)

    def forward(self, x):
        a = auto.addm_backward()
        out = a(x, self.w, self.b)
        out.grad_fn = a
        return out

    def __call__(self, x):
        return self.forward(x)


class ReLU(Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(ReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        fun = auto.relu_backward()
        out = fun(input, self.inplace)
        out.grad_fn = fun
        return out

    def __call__(self, x):
        return self.forward(x)


class Sigmoid(Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(Sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        fun = auto.sigmoid_backward()
        out = fun(input)
        out.grad_fn = fun
        return out

    def __call__(self, x):
        return self.forward(x)


class Softmax(Module):
    def __init__(self, inplace: bool = False):
        super(Softmax, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        fun = auto.softmax_backward()
        out = fun(input)
        out.grad_fn = fun
        return out

    def __call__(self, x):
        return self.forward(x)


class Conv2d(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0):
        super(Conv2d, self).__init__()

        self.w = tensor(np.random.randn(out_channels, in_channels,
                                        kernel_size[0], kernel_size[1]),
                        requires_grad=True,
                        is_leaf=True)

    def forward(self, input):
        fun = auto.conv2d_backward()
        out = fun(input, self.w)
        out.grad_fn = fun
        return out

    def __call__(self, x):
        return self.forward(x)


class Average_pooling(Module):
    def __init__(self, kernel_size):
        super(Average_pooling, self).__init__()
        self.kernel_size = kernel_size

    def __call__(self, x):
        return self.forward(x)

    def forward(self, input):
        fun = auto.average_backward()
        out = fun(input, self.kernel_size)
        out.grad_fn = fun
        return out


class Max_pooling(Module):
    def __init__(self, kernel_size):
        super(Max_pooling, self).__init__()
        self.kernel_size = kernel_size

    def __call__(self, x):
        return self.forward(x)

    def forward(self, input):
        fun = auto.max_backward()
        out = fun(input, self.kernel_size)
        out.grad_fn = fun
        return out


class Dims_change(Module):
    def __init__(self):
        super(Dims_change, self).__init__()

    def __call__(self, x):
        return self.forward(x)

    def forward(self, input):
        fun = auto.DimsBackward()
        out = fun(input)
        out.grad_fn = fun
        return out


def save(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
