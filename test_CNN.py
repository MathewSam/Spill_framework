import numpy as np
import pytest
import math
from main import load_mnist_images
from network import Network
from layers import Linear,BatchNorm,ReLU,Convolution2D,SoftmaxCrossEntropyLoss

def generate_images(batch_size=128):
    '''
    To generate the images needed for purposes of testing
    Args:
        batch_size:(data type:int)size of data needed for testing purposes. Default value at 128
    Returns:
        test_input:(data type:numpy.ndarray)Input samples to be fed into the network for testing
        test_labels:(data type:numpy.ndarray)Labels of input samples fed into the netowrk
    '''
    inputs, labels = load_mnist_images()
    test_input = inputs['train'][:batch_size]
    test_label = labels['train'][:batch_size]
    return test_input,test_label

def generate_CNN():
    '''
    To generate a network with a CNN layer for gradient checking
    Returns:
        net:(data type:Network) Network defined for testing purposes, inthis case specifically for batch norm layer
    '''
    n_classes = 10
    
    net = Network(learning_rate = 1e-3)
    net.add_layer(Convolution2D(1,1,28,28,pad=0,stride=1,filter_size=3,dilation=1))
    net.add_layer(ReLU())
    net.add_layer(BatchNorm(676))
    net.add_layer(Linear(676, 128))
    net.add_layer(ReLU())
    net.add_layer(BatchNorm(128))
    net.add_layer(Linear(128, n_classes))
    net.set_loss(SoftmaxCrossEntropyLoss())
    return net

def generate_weight_gradient(net,grad):
    '''
    Generates gradient for the batch norm layer for the purposes of gradient checking
    Args:
        net:(data_type:Network) predefined network for the purpose of gradient checking
        grad:(data_type:numoy.ndarray)gradient generated at the final layer that need s to be back propagated
    Returns:
        gradient:(data_type:numpy.ndarray)gradient associated with the layer 
    '''
    for layer in reversed(net.layers):
            grad, layer_var_grad = layer.backward(grad)
            if isinstance(layer,Convolution2D):
                return layer_var_grad[0][1]

def generate_bias_gradient(net,grad):
    '''
    Generates gradient for the batch norm layer for the purposes of gradient checking
    Args:
        net:(data_type:Network) predefined network for the purpose of gradient checking
        grad:(data_type:numoy.ndarray)gradient generated at the final layer that need s to be back propagated
    Returns:
        gradient:(data_type:numpy.ndarray)gradient associated with the layer 
    '''
    for layer in reversed(net.layers):
            grad, layer_var_grad = layer.backward(grad)
            if isinstance(layer,Convolution2D):
                return layer_var_grad[1][1]

def weight_eps_plus(net,location,epsilon):
    '''
    Creates a new network to calculate the numeric gradient with a shift of epsilon at a specific location
    Args:
        net:Network defined for testing purposes
        location:Location at which testing takes place
        epsilon: Shift for which testing takes place    n
    '''
    import copy
    net_eps_plus = copy.deepcopy(net) 
    for index,layer in enumerate(net_eps_plus.layers):
        if isinstance(layer,Convolution2D):
            net_eps_plus.layers[index].W[location] = net_eps_plus.layers[index].W[location] +epsilon
            break
    return net_eps_plus

def bias_eps_plus(net,epsilon):
    '''
    Creates a new network to calculate the numeric gradient with a shift of epsilong for the bias
    *assumes only one output channel and has only a single bias value
    Args:
        net:Network defined for testing purposes
        epsilon:Value for which gradient is calculated
    '''
    import copy
    net_eps_plus = copy.deepcopy(net) 
    for index,layer in enumerate(net_eps_plus.layers):
        if isinstance(layer,Convolution2D):
            net_eps_plus.layers[index].b[0,0] = net_eps_plus.layers[index].b[0,0] +epsilon
            break
    return net_eps_plus

@pytest.mark.parametrize("location,epsilon",[
                        ((0,0,0,0),1e-02),
                        ((0,0,0,0),1e-03),
                        ((0,0,0,0),1e-04),
                        ((0,0,1,0),1e-03),
                        ((0,0,0,1),1e-04),
                        ((0,0,1,1),1e-03),
                        ]
                        )
def test_weight_grad_checking(location,epsilon):
    '''
    Carries out numerical gradient checking for positions in the kernel
    Args:
        location:location of weight value in 4D convolution kernel
        epsilon: value of epsilon for which gradient checking is done
    '''
    np.random.seed(42)
    #Generate data for testing
    test_input,test_label = generate_images(batch_size=64)
    #Generate network for testing
    net = generate_CNN()
    #Generate predictions from model
    scores = net.predict(test_input,train=True)
    loss, grad = net.loss.get_loss(scores,test_label)  
    #Generate gradient matrix for batch norm from network(analytical gradient)
    analytic_gradient = generate_weight_gradient(net,grad)
    #Generate new network for numeric gradient calculation
    net_eps_plus = weight_eps_plus(net,location,epsilon)
    scores =  net_eps_plus.predict(test_input)
    loss_eps,_ = net_eps_plus.loss.get_loss(scores,test_label)
    numeric_gradient = (loss_eps - loss)/epsilon  
    print("At location {0} the log of difference between numeric and caculated difference = {1} for an epsilon of {2}\n".format(location,numeric_gradient-analytic_gradient[location],epsilon))
    assert(math.isclose(numeric_gradient,analytic_gradient[location],abs_tol=epsilon,rel_tol=epsilon**2)) 

