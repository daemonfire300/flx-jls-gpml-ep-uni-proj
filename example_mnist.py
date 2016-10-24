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
    training_data, training_labels = mnist.load_mnist("training", digits=[1,7])
    #test_data = expecatation_propagation.getRandomTrainingData(test_data_format) # wrong, please provide meaningful test data
    test_data, test_labels = mnist.load_mnist("testing", digits=[1,7])
    
    # only 1 or -1 are allowed in the binary classifier
    training_labels[training_labels == 7] = -1
    test_labels[test_labels == 7] = -1

    # reshape the data, to have an 2D array
    training_data = training_data.reshape(training_data.shape[0], -1)
    test_data = test_data.reshape(test_data.shape[0], -1)

    # the data array is too big. We have to decrease the size
    size = 10
    training_data = training_data[:size,]
    training_labels = training_labels[:size]
    test_data = test_data[:size,]
    test_labels = test_labels[:size]
    
    
    print("training_data")
    print(training_data.shape)
    print("training_labels")
    print(training_labels.shape)    
    print("test_data_X")
    print(test_data[0].shape)
    print("test_data_y")
    print(test_data[1].shape)
    # learn parameters
    K = kernel.compute(training_data, training_data, 1)
    print("kernel")
    print(K.shape)
    np.set_printoptions(threshold=np.nan)
    print(training_labels)
    v, t = expecatation_propagation.EP_binary_classification(K, training_labels)
    print("v, t")
    print(v, t)
    probability = prediction.classify(v, t, training_data, K, training_labels, kernel.compute, test_data)
    print(probability)

if __name__ == "__main__":
    main()

