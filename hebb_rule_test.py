import numpy as np
import matplotlib.pyplot as plt
import my_optimizer


# Verification of Perceptual Learning Rules Adaptive Associator

def hardline(x):
    for i in range(30):
        x[i][0] = 1 if x[i][0] >= 0 else -1
    return x


def drow(x):
    a = x.reshape([5, 6]).T
    for i in range(6):
        for j in range(5):
            if a[i][j] == -1:
                print(" ", end="")
            if a[i][j] == 1:
                print("#", end="")
        print("")



# sourcery skip: avoid-builtin-shadow
label_0 = np.ones([30, 1])
l = [0, 5, 7, 8, 9, 10, 13, 14, 15, 16, 19, 20, 21, 22, 24, 29]
for i in l:
    label_0[i][0] = -1

label_1 = np.ones([30, 1])
l = [
    0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    28, 29
]
for i in l:
    label_1[i][0] = -1

label_2 = np.ones([30, 1])
l = [1, 2, 3, 4, 5, 7, 8, 13, 14, 16, 18, 21, 22, 24, 25, 26, 27, 28]
for i in l:
    label_2[i][0] = -1

input = np.hstack((label_0, label_1, label_2))
hebb = my_optimizer.Hebb(input, input)
hebb.Step()

label_test = label_0
label_test[1] = -1
label_test[2] = -1
label_test[3] = -1
label_test[17] = -1
label_test[25] = -1
print("Test Image: ")
drow(label_test)
a = hardline(np.dot(hebb.w, label_test))
#绘出来
print("Output:")
drow(a)