1. In this problem, we will implement the backpropagation algorithm and train our first multi-layer perceptron to distinguid between 4 classes. we should implement everything on our own without using existing libraries like Tensorflow, Keras, or PyTorch.


2. we are provided with two files “train data.csv“ and “train labels.csv“ The dataset contains 24754 samples, each with 784 features divided into 4 classes (0,1,2,3). we should divide this into training, and validation sets (a validation set is used to make sure our network did not overfit). we will then provide our model which will be tested with an unseen test set.


3. Use one input layer, one hidden layer, and one output layer in our implementation. The labels are one-hot encoded. For example, class 0 has a label of [1, 0, 0, 0] and class 2 has a label of [0,0,1,0]. Make sure we use the appropriate activation function in the output layer. we are free to use any number of nodes in the hidden layer. we need to provide one single function that allows us to use the network to predict the test set. This function should output the labels one-hot encoded in a numpy array.


4. Our work will be graded based on the the successful implementation of the backpropagation algorithm, the multi-layer perceptron, and the test set accuracy. we may want to thoroughly comment our implementation to allow us to easily understand it.
