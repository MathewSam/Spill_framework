# mnist-numpy
A basic fully connected network implemented purely in NumPy and trained and tested on the MNIST dataset.

## Recommended Python Libraries
1. numpy(ver 1.14.3)
2. math(as in python version 3.6.5)
3. pytest(ver 3.5.1)
4. pickle

## Data Statistics
The MNIST dataset is split into 50000 train, 10000 validation and 10000 test samples. All splits are normalized using the statistics of the training split (using the global mean and standard deviation, not per pixel).

## Experiments

### Vanilla Multi Layer Perceptron
The network has 2 fully connected layers with ReLU activations. The first hidden layer has 256 units and the second 128 units. The network is initialized with Xavier-He initialization.

The network is trained for 250 epochs with vanilla minibatch SGD and learning rate 1e-3. The final accuracy on the test set is about 0.97.

### BatchNorm applied Multilayer Perceptron
The network has 2 fully connected layers with ReLU activations. The first hidden layer has 256 units and the second 128 units with batchnorm operation on data entering and exiting the hidden layer with 128 nuerons. The network is initialized with Xavier-He initialization.

The network is trained for 250 epochs with vanilla minibatch SGD and learning rate 1e-3. The final accuracy on the test set is about 0.973.

### Convolution Neural Networks
The network has a convolutional layer with two 3x3 kernels with a dilation of 2 followed by a hidden layer of 128 neurons which then lead to the output.

The network is trained for 250 epochs with vanilla minibatch SGD and learning rate 1e-2. The final accuracy on the test set is about 0.975.

## Code structure:
### layers.py
Contains classes that represent layers for different transformations. Each class has a forward and a backward method that define a transformation and its gradient. The class keeps track of the variables defining the transformation and the variables needed to calculate the gradient. The file also contains a class that defines the softmax cross entropy loss.

#### Linear
Responsible for the linear transformations between layers
#### ReLU
Responsible for non linear relu operation on inputs to the layer
#### BatchNorm
Responsible for version of batchnorm described in question as in Task1
#### Convolution2D
Responsible for the 2D convolution layer/kernels which is learnt and used as required 

#### SoftmaxCrossEntropyLoss
Uses logits to calculate soft max probabilities and associated loss from cross entropy function


### im2col.py
Introduces a function that calculates the permutation required to convert input image to column vector used to carry out convolution/correlation of the image and kernel using matrix multiplications

### network.py
Defines Network, a configurable class representing a sequential neural network with any combination of layers. Network has a train function that performs minibatch SGD.

### main.py
Data loading, training and validation scripts. Running it trains the networks described in experiments. For loading the data it expects two files "data/mnist_train.csv" and "data/mnist_test.csv". These can be downloaded from https://pjreddie.com/projects/mnist-in-csv/. To run use "python3 main.py".

### test_batch_norm.py
Testing module to test whether the batchnorm operations work

### test_CNN.py
Testing module to test whether the CNN operations work

*Both above test can be run by simply running pytest -v

## Design Choices:
The Convolution2D class is responsible for implementing the convolution layer which replaces the first linear layer. A few design choices were made to speed up the training process. However, with a few small adjustments, one can make the model generalizable to any kind of CNN structure with more than one convolutional layer. 

### Batch Norm:
The mean batch norm for the linear layer is implemented as per the requirements placed in the provided question. On initializing the class , a 1D vector of zeros for the number of features is initialized. This feature vector is the external/ parameterized mean shift that is learnt by the network.For the sake of the problem here, the mean during testing is taken as the running mean of training set.
### Convolution2D :
The current backpropagation algorithm does not backpropagate the signal gradient since the convolutional layer is the first layer without any further convolution layers in the mix. This saves computational energy and time during training making the algorithm train faster. One can adjust this to allow for calculating the signal gradient by replacing the backward and forward function of Convolution2D with


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

    def backward(self,dY):
        
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
        dX = col2im_indices(dX_col, x_shape, self.filter_size, self.filter_size, padding=self.pad, stride=self.stride)
        
        return dX,[(self.W, dW),(self.b, db)]

Using the Vectorize layer can help stack the convolutional layers to the linear layers and provide a means to reshape the gradient as needed. However, making these changes adds time and computation to the code. The current code has been designed to train fast with as little unnecessary computation as possible.

The current code **DOES NOT** use the vectorize class and the vectorize operation is handled internally .The changes allow take the vectorization out of the class. The above changes are suggested to allow design of more complicated and complex networks if desired.

For the current configuration, the model's gradient backpropagation is tested with gradient checking in the test_CNN.py file.


#### Sample architecture(using Vectorize)
A simple architecture with the changes suggested above for more complex networks using Vectorize with gradient backpropagation of the image signal is shown here:

    import numpy as np

    from layers import Linear, ReLU, SoftmaxCrossEntropyLoss,BatchNorm,Convolution2D,Vectorize
    from network import Network

    np.random.seed(42)
    n_classes = 10

    inputs, labels = load_mnist_images()

    net = Network(learning_rate = 1e-3)
    net.add_layer(Convolution2D(1,2,28,28,pad=0,stride=1,filter_size=3,dilation=2))
    net.add_layer(Vectorize())
    net.add_layer(ReLU())
    net.add_layer(BatchNorm(800))
    net.add_layer(Linear(800, 128))
    net.add_layer(ReLU())
    net.add_layer(BatchNorm(128))
    net.add_layer(Linear(128, n_classes))
    net.set_loss(SoftmaxCrossEntropyLoss())

    train_network(net, inputs, labels, 250)
    test_loss, test_acc = validate_network(net, inputs['test'], labels['test'],
                                            batch_size=128)
    print('Baseline MLP Network without batch normalization:')
    print('Test loss:', test_loss)

    print('Test accuracy:', test_acc)

While the above model trains significantly slower, the model gains an accuracy of 0.95 in the first 5 epochs with a learning rate of 1e-2 and vanilla minibacth SGD.

#### Sample architecture(without making the changes explained above):
    
The code shown below is the CNN that is currently being used(without making the changes explained above). As you can probably guess, the vectorization
takes place inside the Cnovolution2D class and the class needs to be changed as explained above to stack more convolutional layers or complicate the network. This current structure allows you to increase the number of convolution kernels you can have in the first convolution layer, while correspondingly changing the number of inputs to the batch norm and the linear layers.\

However, to stack more convolutional layers, the changes explained in the previous section must be implemented.

    np.random.seed(42)
    n_classes = 10

    inputs, labels = load_mnist_images()

    # Define network without batch norm
    net = Network(learning_rate = 1e-3)
    net.add_layer(Convolution2D(1,2,28,28,pad=0,stride=1,filter_size=3,dilation=2))
    net.add_layer(ReLU())
    net.add_layer(BatchNorm(800))
    net.add_layer(Linear(800, 128))
    net.add_layer(ReLU())
    net.add_layer(BatchNorm(128))
    net.add_layer(Linear(128, n_classes))
    net.set_loss(SoftmaxCrossEntropyLoss())

    train_network(net, inputs, labels, 250)
    test_loss, test_acc = validate_network(net, inputs['test'], labels['test'],
                                            batch_size=128)
    print('Baseline CNN Network with batch normalization:')
    print('Test loss:', test_loss)

    print('Test accuracy:', test_acc)
    return net

## Running the code/Usage
To run the code, simply run:
    python main.py network_type -o

The network_types that are supported are:
1. Vanilla (Plain MLP)
2. Batch_Norm (for MLP with batch norm between layers)
3. CNN

Inputing any other type will raise an assertion error. The -o is optional. It enables saving the model as a pickle file. For help, please run:

    python main.py -help