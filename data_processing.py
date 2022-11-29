import struct
import numpy as np
from tensor import tensor


def normalization(ima, inplace=True):
    """
    It takes an image and normalizes it to the range [0,1]
    
    :param ima: the image to be normalized
    :param inplace: if True, the image is normalized in place, otherwise a new image is returned,
    defaults to True (optional)
    :return: The normalized image.
    """
    a_max = np.max(ima)
    a_min = np.min(ima)
    for j in range(ima.shape[0]):
        ima[j] = (ima[j] - a_min) / (a_max - a_min)
    return ima


def rand_select(ls, img, label):
    """
    It takes a list of indices, and a list of images and labels, and returns a single image and label
    that are the concatenation of the images and labels at the indices in the list
    
    :param ls: the list of random numbers
    :param img: the image data
    :param label: the label of the image
    :return: the image and the label.
    """
    z = img[ls[0]]
    o = label[ls[0]]
    for i in range(1, len(ls)):
        #横向拼接
        z = np.hstack((z, img[ls[i]]))
        o = np.hstack((o, label[ls[i]]))
    return z, o


def pre_process(data, label, size):
    """
    It takes the data and label, and normalizes the data
    
    :param data: the data set
    :param label: the label of the image
    :param size: the number of images to be processed
    :return: the normalized image and the label.
    """
    img_train = []
    img_label = []
    for i in range(size):
        img_train.append(normalization(data[i])[None])
        label_temp = np.zeros((10, 1))
        label_temp[int(label[i])] = 1
        img_label.append(label_temp)
    return img_train, img_label



# fill in specified or missing values
def to_fill_in(samples: tensor, to_fill=np.NaN, arg="mean"):
    func = {
        "mean": np.mean,
        "max": np.max,
        "min": np.min,
        "var": np.var,
        "median": np.median
    }

    pos = np.argwhere(samples.data == to_fill)
    for i in range(len(samples.data)):
        syn = pos[:, 0] == i
        lines = pos[syn][:, 1]  # get the column index of the missing value
        tmp = np.delete(samples.data, lines, axis=1)
        args_pre = func[arg](tmp[i])
        samples.data[i, lines] = args_pre

    return samples


# delete the sample where the specified value or missing value is located
def dropn(samples: tensor, to_drop=np.NaN):
    pos = np.argwhere(samples.data == to_drop)
    samples.data = np.delete(samples.data, [i[1] for i in pos], axis=1)
    return samples


def decode_idx3_ubyte_file(idx3_ubyte_file):
    bin_data = open(idx3_ubyte_file, 'rb').read()
    offset = 0
    fmt_header = '>IIII'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(
        fmt_header, bin_data, offset)
    print("magic_number:%d, num_images: %d, num_size: %d*%d" %
          (magic_number, num_images, num_rows, num_cols))

    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = f'>{str(image_size)}B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print("already decode %d" % (i + 1) + "pictures")
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data,
                                                offset)).reshape(
                                                    (num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images


def decode_idx1_ubyte_file(idx1_ubyte_file):
    bin_data = open(idx1_ubyte_file, 'rb').read()

    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print("magic_number:%d, num_images: %d s" % (magic_number, num_images))

    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print("already decode %d" % (i + 1) + "s")
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def load_train_images(idx_ubyte_file):
    return decode_idx3_ubyte_file(idx_ubyte_file)


def load_train_labels(idx_ubyte_file):
    return decode_idx1_ubyte_file(idx_ubyte_file)


def load_test_images(idx_ubyte_file):
    return decode_idx3_ubyte_file(idx_ubyte_file)


def load_test_labels(idx_ubyte_file):
    return decode_idx1_ubyte_file(idx_ubyte_file)


def image2vector(image):
    return np.reshape(image, [784, 1])
