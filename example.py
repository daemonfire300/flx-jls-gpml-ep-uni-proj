# -*- coding: utf-8 -*-
import expecatation_propagation
import prediction
import kernel
import numpy as np

def main():
    training_data, training_labels = expecatation_propagation.getRandomTrainingData(np.zeros((100, 10)))
    test_data = expecatation_propagation.getRandomTrainingData(np.zeros((100, 10))) # wrong, please provide meaningful test data
    # learn parameters
    K = kernel.compute(training_data, training_data, 1)
    v, t = expecatation_propagation.EP_binary_classification(K, training_labels)
    probability = prediction.classify(v, t, X, K, training_labels, kernel.compute, test_data)
    print(probability)

if __name__ == "__main__":
    main()
