import sys
sys.path.append("..")
import utils
from utils import *
import numpy as np
import matplotlib.pyplot as plt

def augmentFeatureVector(X):
    columnOfOnes = np.zeros([len(X), 1]) + 1
    return np.hstack((columnOfOnes, X))


def computeProbabilities(X, theta, tempParameter):
    theta_dot_x = np.divide(np.dot(theta, X.transpose()) , tempParameter)
    c_indices = np.argmax(theta_dot_x, axis=0)
    c_values = np.array([theta_dot_x[c_indices[i]][i] for i in range(theta_dot_x.shape[1])])

    adjusted_values = np.subtract(theta_dot_x, c_values)
    exp_values = np.exp(adjusted_values)
    divisors = np.sum(exp_values, axis=0)
    return np.divide(exp_values, divisors) 


def computeCostFunction(X, Y, theta, lambdaFactor, tempParameter):
    cost = lambdaFactor/2.0*np.sum(np.square(theta))
    probs = computeProbabilities(X, theta, tempParameter)
    n = X.shape[0]
    for i in range(Y.shape[0]):
        cost += -1.0/n * np.log(probs[Y[i]][i])
    return cost


def runGradientDescentIteration(X, Y, theta, alpha, lambdaFactor, tempParameter):
    H = computeProbabilities(X, theta, tempParameter)
    n = X.shape[0]
    M = np.zeros((theta.shape[0], X.shape[0]))
    for i in range(n):
        M[Y[i]][i] = 1 
    gradient = -(1.0/(tempParameter*n)) * np.dot(np.subtract(M,H) , X) + np.multiply(lambdaFactor, theta)
    theta = theta - alpha*gradient
    return theta
    


def updateY(trainY, testY):
    trainYMod3 = trainY % 3
    testYMod3 = testY % 3
    return trainYMod3, testYMod3

def computeTestErrorMod3(X, Y, theta, tempParameter):
    errorCount = 0.
    assignedLabels = getClassification(X, theta, tempParameter)
    predictions, Y = updateY(assignedLabels, Y)
    return 1 - np.sum(np.equal(predictions, Y))/Y.shape[0] 


def softmaxRegression(X, Y, tempParameter, alpha, lambdaFactor, k, numIterations):
    X = augmentFeatureVector(X)
    theta = np.zeros([k, X.shape[1]])
    costFunctionProgression = []
    for i in range(numIterations):
        print("iteration number: ", i)
        costFunctionProgression.append(computeCostFunction(X, Y, theta, lambdaFactor, tempParameter))
        theta = runGradientDescentIteration(X, Y, theta, alpha, lambdaFactor, tempParameter)
    return theta, costFunctionProgression
    
def getClassification(X, theta, tempParameter):
    X = augmentFeatureVector(X)
    probabilities = computeProbabilities(X, theta, tempParameter)
    return np.argmax(probabilities, axis = 0)

def plotCostFunctionOverTime(costFunctionHistory):
    plt.plot(range(len(costFunctionHistory)), costFunctionHistory)
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')
    plt.show()

def computeTestError(X, Y, theta, tempParameter):
    errorCount = 0.
    assignedLabels = getClassification(X, theta, tempParameter)
    return 1 - np.mean(assignedLabels == Y)
