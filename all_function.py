import itertools
import numpy as np
from tensor import *


def relu(relu_input, inplace=False):
    if inplace == True:
        relu_input[relu_input <= 0] = 0
        return relu_input
    else:
        temp = relu_input.data.copy()

        if relu_input.requires_grad == True or relu_input.is_leaf == False:
            return tensor(temp, is_leaf=False, requires_grad=True)
        else:
            return tensor(temp,
                          is_leaf=relu_input.is_leaf,
                          requires_grad=relu_input.requires_grad)


def drelu(drelu_input):
    temp = drelu_input.data.copy()
    temp[temp <= 0] = 0
    temp[temp > 0] = 1
    return tensor(temp,
                  is_leaf=drelu_input.is_leaf,
                  requires_grad=drelu_input.requires_grad)


def sigmoid(sig_input, inplace=False):
    if inplace == True:
        sig_input.data = 1.0 / (1.0 + np.exp(-sig_input.data))
        return sig_input
    else:
        temp = 1.0 / (1.0 + np.exp(-sig_input.data))
        if sig_input.requires_grad == True or sig_input.is_leaf == False:
            return tensor(temp, is_leaf=False, requires_grad=True)
        else:
            return tensor(temp,
                          is_leaf=sig_input.is_leaf,
                          requires_grad=sig_input.requires_grad)


def dsigmoid(dsig_input):
    return sigmoid(dsig_input) * (1 - sigmoid(dsig_input))


def softmax(soft_input):

    temp = np.exp(soft_input)
    out = temp.copy()
    temp = np.sum(temp, axis=0)

    return out / temp


def get_rows(a, w):
    ls = []
    channel = a.shape[0]
    assert (a.shape[0] == w.shape[1])
    for i in range(a.shape[1] - w.shape[2] + 1):
        ls.extend(a[0:channel, i:i + w.shape[2], j:j + w.shape[3]].ravel()
                  for j in range(a.shape[2] - w.shape[3] + 1))

    t_w = np.empty((w[0].size, w.shape[0]))
    for k in range(w.shape[0]):
        w_col = w[k, 0:channel, :, :].flatten()
        t_w[:, k] = w_col
    return np.array(ls), t_w


def anti_cols(y, a_h, a_w, w_h, w_w):
    channel = y.shape[1]
    output = np.empty((channel, a_h - w_h + 1, a_w - w_w + 1))
    for i in range(channel):
        temp = np.reshape(y[:, i], [a_h - w_h + 1, a_w - w_w + 1])
        output[i, :, :] = temp
    return output


def cols(y):
    output = np.empty((y[0].size, y.shape[0]))
    for i in range(y.shape[0]):
        output[:, i] = np.reshape(y[i, :, :], [y[i].size])
    return output


def build_a(a_grad_temp, a, w):
    x_grad = np.zeros_like(a).astype(float)
    c = 0
    for i in range(a.shape[1] - w.shape[2] + 1):
        for j in range(a.shape[2] - w.shape[3] + 1):
            for k in range(0, a.shape[0] * w[0, 0].size, w[0, 0].size):

                x_grad[int(k / w[0, 0].size), i:i + w.shape[2], j:j +
                       w.shape[3]] += a_grad_temp[c,
                                                  k:k + w[0, 0].size].reshape(
                                                      [w.shape[2], w.shape[3]])
            c += 1
    return x_grad


def build_w(w_grad_temp, w_channel, w_high, w_wide):
    num_w = w_grad_temp.shape[1]
    w = np.empty((num_w, w_channel, w_high, w_wide))
    for i in range(num_w):
        w[i, :, :, :] = np.reshape(w_grad_temp[:, i],
                                   [w_channel, w_high, w_wide])
    return w


def Average(x, size):
    assert (len(x.shape) == 3)
    output = np.empty(
        (x.shape[0], int(x.shape[1] / size), int(x.shape[2] / size)))
    for c in range(x.shape[0]):
        for i in range(int(x.shape[1] / size)):
            for j in range(int(x.shape[2] / size)):
                output[c, i, j] = np.sum(x[c, i * size:size *
                                           (i + 1), j * size:size *
                                           (j + 1)]) / size**2
    return output


def dAverage(temp_grad, size, x):
    assert (len(temp_grad.shape) == 3)
    output = np.zeros_like(x).astype(float)
    for c in range(x.shape[0]):
        for i in range(temp_grad.shape[1]):
            for j in range(temp_grad.shape[2]):
                output[c, i * size:size * (i + 1),
                       j * size:size * (j + 1)] = np.full(
                           (size, size), temp_grad[c, i, j] / size**2)
    return output


def Maxpool(x, size):
    assert (len(x.shape) == 3)
    output = np.empty(
        (x.shape[0], int(x.shape[1] / size), int(x.shape[2] / size)))
    record = np.zeros_like(x).astype(float)
    for c in range(x.shape[0]):
        for i in range(int(x.shape[1] / size)):
            for j in range(int(x.shape[2] / size)):
                output[c, i, j] = np.max(x[c, i * size:size * (i + 1),
                                           j * size:size * (j + 1)])
                record[c, i * size:size * (i + 1), j * size:size *
                       (j + 1)][x[c, i * size:size * (i + 1),
                                  j * size:size * (j + 1)] == output[c, i,
                                                                     j]] = 1
    return output, record


def dMaxpool(temp_grad, size, record):

    for c, i in itertools.product(range(record.shape[0]),
                                  range(temp_grad.shape[1])):
        for j in range(temp_grad.shape[2]):
            record[c, i * size:size * (i + 1), j * size:size *
                   (j + 1)][record[c, i * size:size * (i + 1), j * size:size *
                                   (j + 1)] == 1] = temp_grad[c, i, j]
    return record