import numpy as np


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    '''
    Calculates permutation required to convert incoming image to desired column vector to implement 2D convolution as a matrix multiplication
    Args:
        x_shape:original shape of input image
        field_height:height of kernel used for convolution
        field_width: width of kernel used for convolution
    Kwargs:
        padding:(data type:int)padding yet to be introduced to the image(default value:0)
        stride:(data type:int)stride used while traversing the image(default value:1)
    '''
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k.astype(int), i.astype(int), j.astype(int))


