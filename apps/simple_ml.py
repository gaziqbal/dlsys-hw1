import gzip
import os
import struct
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../python/"))
import needle as ndl


def parse_mnist(image_filesname, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    # read images
    with gzip.open(image_filesname, "rb") as image_file:
        header_bytes = image_file.read(16)
        header, num_images, num_rows, num_columns = struct.unpack(
            '>Iiii', header_bytes)
        if header != 0x00000803:
            raise Exception("invalid header for image file")
        image_size = num_rows * num_columns
        dataset_size = num_images * image_size
        image_data = np.frombuffer(
            image_file.read(dataset_size), dtype=np.uint8)
        # normalize
        min = np.min(image_data)
        max = np.max(image_data)
        image_data = (image_data - min) / (max - min)
        # shape
        images = np.reshape(image_data.astype(
            np.float32), (num_images, image_size))

    # read labels
    with gzip.open(label_filename, "rb") as label_file:
        header_bytes = label_file.read(8)
        header, num_labels = struct.unpack('>Ii', header_bytes)
        if header != 0x00000801:
            raise Exception("invalid header for label file")
        labels = np.frombuffer(label_file.read(num_labels), dtype=np.uint8)

    return images, labels
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    z_exp = ndl.exp(Z)
    z_sum = ndl.summation(z_exp, axes=(1,))
    z_log = ndl.log(z_sum)
    z_y = ndl.summation(ndl.multiply(Z, y_one_hot), axes=(1,))
    z_diff = z_log - z_y
    z_diff_sum = ndl.summation(z_diff, axes=(0,))
    softmax = ndl.divide_scalar(z_diff_sum, Z.shape[0])
    return softmax
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h,y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
