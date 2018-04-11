import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
def loadmnist():
    """
    Loads mnist images and one hot representation labels

    Args:
        None

    Returns:
        mnist: Object containing images and labels
    """
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    return mnist


def plotdigit(digitnr):
    """ Plots 
    """


a=loadmnist()


