import numpy as np
from im2col import *

class Layer(object):
    '''
    Abstract class representing a neural network layer
    '''
    def forward(self, X, train=True):
        '''
        Calculates a forward pass through the layer.

        Args:
            X (numpy.ndarray): Input to the layer with dimensions (batch_size, input_size)

        Returns:
            (numpy.ndarray): Output of the layer with dimensions (batch_size, output_size)
        '''
        raise NotImplementedError('This is an abstract class')

    def backward(self, dY):
        '''
        Calculates a backward pass through the layer.

        Args:
            dY (numpy.ndarray): The gradient of the output with dimensions (batch_size, output_size)

        Returns:
            dX, var_grad_list
            dX (numpy.ndarray): Gradient of the input (batch_size, output_size)
            var_grad_list (list): List of tuples in the form (variable_pointer, variable_grad)
                where variable_pointer and variable_grad are the pointer to an internal
                variable of the layer and the corresponding gradient of the variable
        '''
        raise NotImplementedError('This is an abstract class')

class Linear(Layer):
    def __init__(self, input_dim, output_dim):
        '''
        Represent a linear transformation Y = X*W + b
            X is an numpy.ndarray with shape (batch_size, input_dim)
            W is a trainable matrix with dimensions (input_dim, output_dim)
            b is a bias with dimensions (1, output_dim)
            Y is an numpy.ndarray with shape (batch_size, output_dim)

        W is initialized with Xavier-He initialization
        b is initialized to zero
        '''
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0/input_dim)
        self.b = np.zeros((1, output_dim))
        self.cache_in = None

    def __repr__(self):
        return "Linear Layer"

    def forward(self, X, train=True):
        out = np.matmul(X, self.W) + self.b
        if train:
            self.cache_in = X
        return out

    def backward(self, dY):
        if self.cache_in is None:
            raise RuntimeError('Gradient cache not defined. When training the train argument must be set to true in the forward pass.')
        db = np.sum(dY, axis=0, keepdims=True)
        dW = np.matmul(self.cache_in.T, dY)
        dX = np.matmul(dY, self.W.T)
        return dX, [(self.W, dW), (self.b, db)]

class ReLU(Layer):
    def __init__(self):
        '''
        Represents a rectified linear unit (ReLU)
            ReLU(x) = max(x, 0)
        '''
        self.cache_in = None
    
    def __repr__(self):
        return "Relu Non Linearity"

    def forward(self, X, train=True):
        if train:
            self.cache_in = X
        return np.maximum(X, 0)

    def backward(self, dY):
        if self.cache_in is None:
            raise RuntimeError('Gradient cache not defined. When training the train argument must be set to true in the forward pass.')
        return dY * (self.cache_in >= 0), []

class BatchNorm(Layer):
    def __init__(self,num_features):
        '''
        Carries out layer normalization given output dimensions
        Args:
            num_features:number of features in vector for which normalization is done
        Returns:
            out:cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1) mean normalized and shifted features
        '''
        self.beta = np.zeros((1,num_features))
    
    def __repr__(self):
        return "Batch Normalization"

    def forward(self,X,train=True):
        self.weight = np.eye(X.shape[0]) - (np.ones((1,X.shape[0]))/X.shape[0])
        X_norm = X - np.mean(X,axis = 0)
        X_hat = X_norm + self.beta
        return X_hat
    
    def backward(self,dY):
        dbeta = np.sum(dY, axis=0, keepdims=True)
        dX = np.matmul(self.weight.T,dY)
        return dX, [(self.beta,dbeta)]

class Convolution2D(Layer):
    '''
    Creates and operates kernels for convolution/correlation operations on 2D images
    '''
    def __init__(self,channels_in,channels_out,image_height,image_width,pad=0,stride=1,filter_size=3,dilation=1):
        '''
        Args:
            self:pointer to current instance of the class
            channels_in:(data type:int)number of channels of samples entering the class/depth of input feature map
            channels_out:(data type:int)number of output channels exiting after CNN
            image_height:(data type:int)Height of input image
            image_width:(data type:int)Width of input image
        Kwargs:
            pad:(data type:int) default value 0
            stride:(data type:int) default value 1
            filter_size:(data type:int)default value 3
            dilation:(data type:int)default value 1
        '''
        assert (image_height - filter_size + 2*pad)% stride ==0
        assert (image_width - filter_size + 2*pad)% stride ==0

        #Image features
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.image_height = image_height
        self.image_width = image_width

        #Filter features
        self.pad = pad
        self.stride = stride
        self.filter_size = filter_size**dilation

        #Features for backprop 
        self.cache_in = None

        #Weight Initializations
        self.W = np.random.randn(channels_out,channels_in,self.filter_size,self.filter_size)/np.sqrt(self.filter_size)
        self.b = np.zeros((channels_out,1))

        self.set_weights_zero = np.ones_like(self.W)
        for i in range(filter_size):
            if i%dilation!=0:
                self.set_weights_zero[:,:,i,:] = 0
                self.set_weights_zero[:,:,:,i] = 0
        self.W = self.W*self.set_weights_zero

        #calculating shuffling caused by im2col
        self.x_shape = (None,channels_in,image_height,image_width)
        self.channel_shuffling,self.row_shuffling,self.column_shuffling = get_im2col_indices(self.x_shape, self.filter_size, self.filter_size, padding=self.pad, stride=self.stride) 
        
    def __repr__(self):
        return "2DConvolution layer"

    def forward(self,X,train=True):
        batch_size,channels_in,image_height,image_width = X.shape
        self.output_height = (image_height - self.filter_size + 2*self.pad)//self.stride + 1
        self.output_width = (image_width - self.filter_size + 2*self.pad)//self.stride + 1
        
        X_padded = np.pad(X, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)), mode='constant')
        temp_Weights = self.W.reshape(self.channels_out,-1)

        cols = X_padded[:, self.channel_shuffling, self.row_shuffling, self.column_shuffling]
        cols = cols.transpose(1, 2, 0).reshape((self.filter_size) * (self.filter_size) * channels_in, -1)

        if(train==True):
            self.cache_in = cols

        output = np.dot(temp_Weights,cols) + self.b
        output = output.reshape(self.channels_out,self.output_height,self.output_width,batch_size)
        output = output.transpose(3, 0, 1, 2)
        return output
        #return output.reshape(batch_size,-1)

    def col2im_indices(self,cols, x_shape, field_height=3, field_width=3, padding=1,stride=1):
        N, C, H, W = x_shape
        H_padded, W_padded = H + 2 * padding, W + 2 * padding
        x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
        k,i,j = self.channel_shuffling,self.row_shuffling,self.column_shuffling
        cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
        cols_reshaped = cols_reshaped.transpose(2, 0, 1)
        np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
        if padding == 0:
            return x_padded
        return x_padded[:, :, padding:-padding, padding:-padding]


    def backward(self,dY):
        
        #reshaping incoming gradients to output image dimensions
        #dY = dY.reshape(-1,self.channels_out,self.output_height,self.output_width)
        batch_size = dY.shape[0]
        #Calculating bias
        db = np.sum(dY, axis=(0, 2, 3))
        db = db.reshape(self.channels_out, 1)
        
        #Calculating weights in kernel
        dY_reshaped = dY.transpose(1, 2, 3, 0).reshape(self.channels_out, -1)
        dW = np.dot(dY_reshaped,self.cache_in.T)
        dW = dW.reshape(self.W.shape)
        #Setting dilation positions to 0
        dW = dW*self.set_weights_zero

        #Calculating X gradient for future use
        W_reshape = self.W.reshape(self.channels_out, -1)
        dX_col = np.dot(W_reshape.T,dY_reshaped)
        x_shape = (batch_size,self.x_shape[1],self.x_shape[2],self.x_shape[3]) 
        dX = self.col2im_indices(dX_col, x_shape, self.filter_size, self.filter_size, padding=self.pad, stride=self.stride)

        return dX,[(self.W, dW),(self.b, db)]


class Vectorize(Layer):
    def __init__(self):
        '''
        Vectorizes incoming data to shape batch_size,num_features
        '''
        self.cache_in = None
    
    def __repr__(self):
        return "Vectorization function"

    def forward(self, X, train=True):
        if train:
            self.cache_in = (X.shape[1],X.shape[2],X.shape[3])
        batch_size = X.shape[0]
        return X.reshape(batch_size,-1)

    def backward(self, dY):
        if self.cache_in is None:
            raise RuntimeError('Gradient cache not defined. When training the train argument must be set to true in the forward pass.')
        return dY.reshape(-1,self.cache_in[0],self.cache_in[1],self.cache_in[2]), []


class Loss(object):
    '''
    Abstract class representing a loss function
    '''
    def get_loss(self):
        raise NotImplementedError('This is an abstract class')

class SoftmaxCrossEntropyLoss(Loss):
    '''
    Represents the categorical softmax cross entropy loss
    '''

    def get_loss(self, scores, labels):
        '''
        Calculates the average categorical softmax cross entropy loss.

        Args:
            scores (numpy.ndarray): Unnormalized logit class scores. Shape (batch_size, num_classes)
            labels (numpy.ndarray): True labels represented as ints (eg. 2 represents the third class). Shape (batch_size)

        Returns:
            loss, grad
            loss (float): The average cross entropy between labels and the softmax normalization of scores
            grad (numpy.ndarray): Gradient for scores with respect to the loss. Shape (batch_size, num_classes)
        '''
        scores_norm = scores - np.max(scores, axis=1, keepdims=True)
        scores_norm = np.exp(scores_norm)
        scores_norm = scores_norm / np.sum(scores_norm, axis=1, keepdims=True)

        true_class_scores = scores_norm[np.arange(len(labels)), labels]
        loss = np.mean(-np.log(true_class_scores))

        one_hot = np.zeros(scores.shape)
        one_hot[np.arange(len(labels)), labels] = 1.0
        grad = (scores_norm - one_hot) / len(labels)

        return loss, grad

