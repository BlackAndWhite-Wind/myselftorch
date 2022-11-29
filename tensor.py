import numpy as np

def multiple_matrix(x, y):
    """
    If both x and y are vectors, then return the element-wise product of x and y. If both x and y are
    matrices, then return the matrix product of x and y. If x is a vector and y is a matrix, then return
    the matrix product of x and y. If x is a matrix and y is a vector, then return the matrix product of
    x and y
    
    :param x: the first tensor
    :param y: the output of the function
    :return: A tensor object
    """
    if len(x.shape) == 1 or len(y.shape) == 1:
        return tensor(x.data * y.data,
                      is_leaf=True,
                      requires_grad=False,
                      grad_fn='Mul')
    elif x.requires_grad == False and y.requires_grad == False:
        return tensor(np.dot(x.data, y.data),
                      is_leaf=True,
                      requires_grad=False,
                      grad_fn='Mul')

    else:
        return tensor(np.dot(x.data, y.data),
                      is_leaf=False,
                      requires_grad=True,
                      grad_fn='Mul')

# The class is a wrapper around a numpy array. It has a bunch of methods that allow you to do
# operations on the array.
class tensor():
    def __init__(self,
                 x,
                 requires_grad=False,
                 is_leaf=True,
                 grad_fn=None) -> None:
        if isinstance(x, np.ndarray):
            self.data = x
        elif isinstance(x, int):
            self.data = np.array(float(x))
        else:
            self.data = np.array(x)
        self.shape = self.data.shape
        self.grad = None
        self.requires_grad = requires_grad
        self.is_leaf = is_leaf
        self.temp_grad = None
        self.grad_fn = grad_fn

    def trans(self):
        return tensor(self.data.T, is_leaf=True)

    def __str__(self):
        return f"tensor {self.data} requires_grad={self.requires_grad}"

    def __add__(self, rhs):
        """
        If the tensor is a leaf node, then it's a constant and we don't need to track it. If it's not a
        leaf node, then we need to track it
        
        :param rhs: the right hand side of the addition
        :return: A tensor object with the data being the sum of the two tensors.
        """
        if isinstance(rhs, (int, float)):
            return tensor(self.data + rhs,
                          is_leaf=True,
                          requires_grad=False,
                          grad_fn='Add')
        if self.requires_grad == False and rhs.requires_grad == False:
            return tensor(self.data + rhs.data,
                          is_leaf=True,
                          requires_grad=False,
                          grad_fn='Add')
        else:
            return tensor(self.data + rhs.data,
                          is_leaf=False,
                          requires_grad=True,
                          grad_fn='Add')

    def __radd__(self, lhs):
        return self.__add__(lhs)

    def __iadd__(self, rhs):
        return self.__add__(rhs)

    def __sub__(self, rhs):
        if isinstance(rhs, (int, float)):
            return tensor(self.data - rhs,
                          is_leaf=True,
                          requires_grad=False,
                          grad_fn='Sub')
        if self.requires_grad == False and rhs.requires_grad == False:
            return tensor(self.data - rhs.data,
                          is_leaf=True,
                          requires_grad=False,
                          grad_fn='Sub')
        else:
            return tensor(self.data - rhs.data,
                          is_leaf=False,
                          requires_grad=True,
                          grad_fn='Sub')

    def __rsub__(self, lhs):
        return -self.__sub__(lhs)

    def __mul__(self, rhs):
        if isinstance(rhs, (int, float)):
            return tensor(self.data * rhs,
                          is_leaf=self.is_leaf,
                          requires_grad=self.requires_grad,
                          grad_fn='Mul')
        elif self.requires_grad == False and rhs.requires_grad == False:
            return tensor(self.data * rhs.data,
                          is_leaf=True,
                          requires_grad=False,
                          grad_fn='Mul')
        else:
            return tensor(self.data * rhs.data,
                          is_leaf=False,
                          requires_grad=True,
                          grad_fn='Mul')

    def __rmul__(self, other):
        return self.__mul__(other)

    def __getitem__(self, i):
        return tensor(self.data[i],
                      is_leaf=self.is_leaf,
                      requires_grad=self.requires_grad)

    def __lt__(self, rhs):
        return tensor(self.data < rhs,
                      is_leaf=self.is_leaf,
                      requires_grad=self.requires_grad)

    def __le__(self, rhs):
        return tensor(self.data <= rhs,
                      is_leaf=self.is_leaf,
                      requires_grad=self.requires_grad)

    def __gt__(self, rhs):
        return tensor(self.data > rhs,
                      is_leaf=self.is_leaf,
                      requires_grad=self.requires_grad)

    def __ge__(self, rhs):
        return tensor(self.data >= rhs,
                      is_leaf=self.is_leaf,
                      requires_grad=self.requires_grad)

    def __eq__(self, rhs):
        return tensor(self.data == rhs,
                      is_leaf=self.is_leaf,
                      requires_grad=self.requires_grad)

    def __ne__(self, rhs):
        return tensor(self.data != rhs,
                      is_leaf=self.is_leaf,
                      requires_grad=self.requires_grad)

    def __setitem__(self, i, x):
        if isinstance(i, tensor) and isinstance(x, int):
            self.data[i.data] = x
        elif isinstance(x, int):
            self.data[i] = x
        else:
            self.data[i] = x.data

    def __neg__(self):
        return tensor(-self.data,
                      is_leaf=self.is_leaf,
                      requires_grad=self.requires_grad)

    def backward(self, grad_output=None):
        """
        The function is used to calculate the gradient of the output tensor with respect to the input tensor
        
        :param grad_output: The gradient of the loss function with respect to the output of the current node
        """
        if self.grad_fn is None or self.is_leaf == True:
            AttributeError("Your output may be leaf!Please check it.")
        if grad_output is None:
            AttributeError(
                "Your output is not a scalar,but you didn't provide the tensor!"
            )
        self.grad_fn.backward(grad_output)



