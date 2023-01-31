## Troy Williams
#  Pattern Recognition Programming Assignment 1
#  Bayesian MNIST Digit Classification

# Import libraries
import sys
import math
import numpy as np
import pandas as pd
import seaborn as sn
import sklearn.metrics as met
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from mlxtend.data import loadlocal_mnist

# A Bayes classifier trained on m and tested on n MNIST digit images 
def Bayes(train_im, train_labels, test_im, test_labels):

	# Find the prior probabilities for the possible classes according to the training data
	(priors, hist) = prior_probabilities(train_labels)

	# Find the maximum likelihood μ and covariance matrix given the training data
	(μs, Σs) = maximum_log_likelihood(train_im, train_labels)

	# Make predictions for each of the test images
	preds = [Bayes_Predict(im, priors, μs, Σs) for im in test_im]

	# Return the results
	return (preds, test_labels)

# Implements Bayes Decision rule: (Vi,j p(x|wi)p(wi) > p(x|wj)p(wj)) => (x <- wi)  
def Bayes_Predict(image, priors, μs, Σs):

	# Compute the Discrimnant function for the input image given each class
	activation = [Bayes_Discrimant_Function(image, μs[i], Σs[i], priors[i]) for i in range(0,9)]

	# Return the class yielding the maximum activation
	return activation.index(max(activation))

# Bayes Descrimnant Function for the Multivatiate Gaussian Density given arbitraty μ and Σ
def Bayes_Discrimant_Function(x, μ, Σ, p_ω):

	# Compute the psudoinverse of Σ
	inv_Σ = np.linalg.pinv(Σ)

	# Compute discrimnant function
	Wi = -.5 * inv_Σ
	wi = inv_Σ.dot(μ)
	wi0 = (-.5 * np.transpose(μ).dot(inv_Σ.dot(μ))) - (.5 * np.log(np.exp(np.trace(Σ)))) + np.log(p_ω)

	# Return the solution
	return (np.transpose(x).dot(Wi.dot(x))) + (np.transpose(wi).dot(x)) + wi0

# Computes the maximum log liklihood estimates of μ and Σ given the labeled training set
def maximum_log_likelihood(images, labels):
	
	# Make lists to store means and variances 
	mu_hats = []
	sigma_hats = []

	# Iterate over the classes
	for i in range(0,9) :

		# Get subset of data from class i
	
		#  Estimate μ and Σ for the subset
	
	# Return them
	return (mu_hats, sigma_hats)

# Trains a classifier of the specified type on the MNIST digit dataset
def MNIST_classifier(type, k, train_im, train_labels, test_im, test_labels):

	# Load an norm data
	train_im, train_labels = loadlocal_mnist(images_path='train-images-idx3-ubyte', labels_path='train-labels-idx1-ubyte')
	test_im, test_labels = loadlocal_mnist(images_path='t10k-images-idx3-ubyte', labels_path='t10k-labels-idx1-ubyte')
	train_im = train_im/255
	test_im = test_im/255

	# Select the appropriate classifier and return its predictions given the m samples from the MNIST training set and n samples from the MNIST test set
	print("Classifier: " + t + " |Training Set| = " + str(n) + " |Testing Set| = " + str(m) + " k = " + str(k))
	if(t == "Bayes") : results.append(Bayes(train_im[0:n], train_labels[0:n], test_im[0:m], test_labels[0:m]))
	elif(t == "KNN") : results.append(kNN(k, train_im[0:n], train_labels[0:n], test_im[0:m], test_labels[0:m]))

	# Unpack and plot the results
	a, b = cm
	Confusion_Matrix(a, b, [0,1,2,3,4,5,6,7,8,9])

# Runs classification with parameters defined in cases
def MNIST_experiment(cases):

	# Load an norm data
	train_im, train_labels = loadlocal_mnist(images_path='train-images-idx3-ubyte', labels_path='train-labels-idx1-ubyte')
	test_im, test_labels = loadlocal_mnist(images_path='t10k-images-idx3-ubyte', labels_path='t10k-labels-idx1-ubyte')
	train_im = train_im/255
	test_im = test_im/255
	
	# Run classification for each test case, storing the results
	results = []
	for (t, n, m, k) in cases : 
		print("CASE: " + t + " |Training Set| = " + str(n) + " |Testing Set| = " + str(m) + " k = " + str(k))
		if(t == "Bayes") : results.append(Bayes(train_im[0:n], train_labels[0:n], test_im[0:m], test_labels[0:m]))
		elif(t == "kNN") : results.append(kNN(k, train_im[0:n], train_labels[0:n], test_im[0:m], test_labels[0:m]))
	
	# Plot Confusion Matricies and print accuracy for each of the input cases
	for pred, true in results : Confusion_Matrix(pred, true, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# Compute the prior probabilities of the classes given the training labels
def prior_probabilities(labels):

	
    
    return (norm , priors)

# Calculates and plots the confusion matrix
def Confusion_Matrix(y_pred, y_true, y_classes):

	# Make the confusion matrix given the input data
	
	# Make a graphic from the confusion matrix
	
	# Get correct predictions
	
	# Get the histogram for the test set
	
	# Accuracy = num correct / num existing in dataset  print(correct/hist)
	
	# Display the confusion matrix
	
# Run classification with the given parameters, m <= 59999 n <= 9999
