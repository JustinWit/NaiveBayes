# naivebayes.py
"""Perform document classification using a Naive Bayes model."""

import argparse
import os
import pdb
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

ROOT = '~/Documents/FSC/4-Senior/CSC3520/code/csc3520_hw1_f22/data'  # change to path where data is stored

parser = argparse.ArgumentParser(description="Use a Naive Bayes model to classify text documents.")
parser.add_argument('-x', '--training_data',
                    help='path to training data file, defaults to ROOT/trainingdata.txt',
                    default=os.path.join(ROOT, 'trainingdata.txt'))
parser.add_argument('-y', '--training_labels',
                    help='path to training labels file, defaults to ROOT/traininglabels.txt',
                    default=os.path.join(ROOT, 'traininglabels.txt'))
parser.add_argument('-xt', '--testing_data',
                    help='path to testing data file, defaults to ROOT/testingdata.txt',
                    default=os.path.join(ROOT, 'testingdata.txt'))
parser.add_argument('-yt', '--testing_labels',
                    help='path to testing labels file, defaults to ROOT/testinglabels.txt',
                    default=os.path.join(ROOT, 'testinglabels.txt'))
parser.add_argument('-n', '--newsgroups',
                    help='path to newsgroups file, defaults to ROOT/newsgroups.txt',
                    default=os.path.join(ROOT, 'newsgroups.txt'))
parser.add_argument('-v', '--vocabulary',
                    help='path to vocabulary file, defaults to ROOT/vocabulary.txt',
                    default=os.path.join(ROOT, 'vocabulary.txt'))


def main(args):
    print("Document Classification using Naïve Bayes Classifiers")
    print("=======================")
    print("PRE-PROCESSING")
    print("=======================")

    # Parse input arguments
    training_data_path = os.path.expanduser(args.training_data)
    training_labels_path = os.path.expanduser(args.training_labels)
    testing_data_path = os.path.expanduser(args.testing_data)
    testing_labels_path = os.path.expanduser(args.testing_labels)
    newsgroups_path = os.path.expanduser(args.newsgroups)
    vocabulary_path = os.path.expanduser(args.vocabulary)

    # Load data from relevant files
    print("Loading training data...")  
    xtrain = np.loadtxt(training_data_path, int)  # creates a 2-dim array of arrays of size 3
    print("Loading training labels...")

    ytrain = np.loadtxt(training_labels_path, int)  # creates 1-dim array
    print("Loading testing data...")
    xtest = np.loadtxt(testing_data_path, int)  # creates a 2-dim array of arrays of size 3

    print("Loading testing labels...")
    ytest = np.loadtxt(testing_labels_path, int)  # creates 1-dim array

    print("Loading newsgroups...")
    newsgroups = np.loadtxt(newsgroups_path, str)  # creates 1-dim array

    print("Loading vocabulary...")
    vocabulary = np.loadtxt(vocabulary_path, str)  # creates 1-dim array

    # Change 1-indexing to 0-indexing for labels, docID, wordID
    xtrain[:, 0:2] = xtrain[:, 0:2] - 1
    xtest[:, 0:2] = xtest[:, 0:2] - 1
    ytrain = ytrain - 1
    ytest = ytest - 1


    # Extract useful parameters
    num_training_documents = len(ytrain)
    num_testing_documents = len(ytest)
    num_words = len(vocabulary)
    num_newsgroups = len(newsgroups)

    print("\n=======================")
    print("TRAINING")
    print("=======================")

    # Estimate the prior probabilities
    print("Estimating prior probabilities via MLE...")
    # P(document / total num documents)
    priors = np.bincount(ytrain) * (1/num_training_documents)

    # Estimate the class conditional probabilities
    print("Estimating class conditional probabilities via MAP...")
    # probabiliy of word given of type documents
    class_conditionals = np.zeros((num_newsgroups, num_words))  # each row is a different doc type with num_word indecies to track counts
    
    # loop over doc types
    for x in range(num_newsgroups):
        # create a mask to find each document of given doc type
        mask = (ytrain[xtrain[:, 0]] == x)
        # loop over those documents adding them to count totals
        for y in xtrain[mask]:
            class_conditionals[x, y[1]] += y[2]
    
    # add beta
    beta = 1 / num_words
    class_conditionals = class_conditionals + beta

    # divide each row by row sum
    for x in range(num_newsgroups):
        class_conditionals[x] = class_conditionals[x] * (1 / (np.sum(class_conditionals[x])))


    print("\n=======================")
    print("TESTING")
    print("=======================")

    # Test the Naive Bayes classifier
    print("Applying natural log to prevent underflow...")
    log_priors = np.log(priors)
    log_class_conditionals = np.transpose(np.log(class_conditionals))


    print("Counting words in each document...")
    counts = np.zeros((num_testing_documents, num_words))
    for x in xtest:
        counts[x[0], x[1]] += x[2]  # counts[docType, wordPos] += wordCount

    print("Computing posterior probabilities...")
    log_posterior = np.matmul(counts, log_class_conditionals) + log_priors

    print("Assigning predictions via argmax...")
    pred = np.argmax(log_posterior, axis=1)

    print("\n=======================")
    print("PERFORMANCE METRICS")
    print("=======================")

    # Compute performance metrics
    accuracy = np.bincount(ytest[:]==pred)
    accuracy = accuracy[1] / np.sum(accuracy)
    print("Accuracy: {}".format(accuracy))

    cm = confusion_matrix(ytest, pred)
    print("Confusion matrix:")
    for x in cm:
        for y in x:
            print(f"{y:4}", end=" ")
        print()

    # pdb.set_trace()  # uncomment for debugging, if needed


if __name__ == '__main__':
    main(parser.parse_args())
