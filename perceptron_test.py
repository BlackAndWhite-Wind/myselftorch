# Perceptual Learning Rules -- Fruit Classification
import my_optimizer
import numpy as np

# 
points = np.array([[1, 1], [-1, 1], [-1, -1]])
targets = np.array([[0, 1]])
w_init = np.array([[0.5, -1, -0.5]])
b_init = np.array([[0.5]])

optimizer = my_optimizer.Perceptron(points, targets, w_init, b_init)

def get_weight_and_bias(optimizer):
    """
    If the value of w and b is not changed in this update, the training will be terminated
    
    :param optimizer: The optimizer object
    :return: The weight and bias of the last iteration
    """
    for _ in range(20):
        origin_w, origin_b = optimizer.get_w().copy(), optimizer.get_b().copy()
        optimizer.Step()
    # If the value of w and b is not changed in this update, the training will be terminated
        if (origin_w == optimizer.get_w()).all() and (origin_b == optimizer.get_b()).all():
            print(optimizer.get_w(), optimizer.get_b())
            break
    return origin_w,origin_b

origin_w, origin_b = get_weight_and_bias(optimizer) # Get the weight and bias of the last iteration

output = np.dot(origin_w, points) + origin_b
output[output < 0] = 0
output[output > 0] = 1

# Now the w and b can separate two column vectors
print(f"Column vector: {points [:, 0]} is labeled {output [:, 0]}")
print(f"Column vector: {points [:, 1]} is labeled {output [:, 1]}")
