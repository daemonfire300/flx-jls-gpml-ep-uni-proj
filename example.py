# -*- coding: utf-8 -*-
import expecatation_propagation
import prediction
import kernel
import numpy as np
import mnist

def main():
    #test_data_format = np.zeros((100, 10))
    #print(test_data_format.shape)
    #training_data, training_labels = expecatation_propagation.getRandomTrainingData(test_data_format)
    training_data, training_labels = mnist.load_mnist("training")
    print("training_data")
    print(training_data.shape)
    print("training_labels")
    print(training_labels.shape)
    #test_data = expecatation_propagation.getRandomTrainingData(test_data_format) # wrong, please provide meaningful test data
    test_data, test_labels = mnist.load_mnist("training")
    print("test_data_X")
    print(test_data[0].shape)
    print("test_data_y")
    print(test_data[1].shape)
    # learn parameters
    K = kernel.compute(training_data, training_data, 1) # Calling this with the MNIST Data throws: ValueError: XA must be a 2-dimensional array.
    print("kernel")
    print(K.shape)
    v, t = expecatation_propagation.EP_binary_classification(K, training_labels)
    print("v, t")
    print(v)
    print(t)
    probability = prediction.classify(v, t, training_data, K, training_labels, kernel.compute, test_data)
    print(probability)

if __name__ == "__main__":
    main()
