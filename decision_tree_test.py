# Script to train and test the Decision Tree

import sys
import getopt
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from fetchdata import load_cols_train_test, csv_attributes, load_cols
from utils import print_cm, print_parameters, lines_to_list
import pydotplus
import numpy as np
from utils import cross_validation_datasets

def print_results(train_results, test_results, labels):
    (train_acc, train_report, train_cm) = train_results
    (test_acc, test_report, test_cm) = test_results
    print('On training set')
    print('Accuracy: %f' % train_acc)
    print(train_report)
    print('Confusion matrix')
    print_cm(train_cm, labels)
    print()
    print('On test set')
    print('Accuracy: %f' % test_acc)
    print(test_report)
    print('Confusion matrix')
    print_cm(test_cm, labels)
    print()

def features_names_pad(csv_filename, pad_prev):
    prev_cur = ['Prev', 'Cur']
    names = csv_attributes(csv_filename)[:-1]   #skip the class
    if not pad_prev:
        return names
    else:
        return [e[1]+str(prev_cur[e[0] // len(names)]) for e in enumerate(names + names)]

def print_pdf(classifier, filename, features_names, labels):
    'Print the decision tree on pdf'
    dot_data = tree.export_graphviz(classifier,
                                    out_file=None,
                                    feature_names= features_names,  
                                    class_names=labels) 
    graph = pydotplus.graph_from_dot_data(dot_data) 
    graph.write_pdf(filename)

def decision_tree_classify(trainData, trainLabels, testData, testLabels, crit, min_split, max_depth):
    classifier = tree.DecisionTreeClassifier(criterion=crit,min_samples_split=min_split,max_depth=max_depth)
    classifier.fit(trainData, trainLabels)
    trainPred = classifier.predict(trainData)
    predicted = classifier.predict(testData)

    train_acc = accuracy_score(trainLabels, trainPred)
    train_report = classification_report(trainLabels, trainPred)
    train_cm = confusion_matrix(trainLabels,trainPred)

    test_acc = accuracy_score(testLabels, predicted)
    test_report = classification_report(testLabels, predicted)
    test_cm = confusion_matrix(testLabels,predicted)

    return classifier, (train_acc, train_report, train_cm), (test_acc, test_report, test_cm)

def main():
    #---------------------------Parameters-------------------------------------
    basedir = '' #'../crunched_data/' (for example)
    specified_basedir = False
    filenames =  ['233_day.csv','239_day.csv']
    pdffile = None
    labels = ['Awake','Nrem','Rem']
    crit = 'gini'
    min_split = 5
    max_depth = 5
    test_provided = False
    test_filenames = []
    pad_prev = False
    split = True
    N = -1
    cross_validate_individuals = False

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'f:o:b:pcl:t:N:h')
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)
    for o, a in opts:
        if o == '-f':
            filenames = a.split(',')
        if o == '-t':
            test_provided = True
            test_filenames = a.split(',')
        if o == '-o':
            pdffile = a
        if o == '-p':
            pad_prev = True
        if o == '-l':
            filenames = lines_to_list(a)
        if o == '-b':
            specified_basedir = True
            basedir = a
        if o == '-N':
            N = int(a)
            split = False
        if o == '-c':
            cross_validate_individuals = True
        elif o == '-h':
            print('''USAGE: python decision_tree_test.py [-f <filenames>] [-t <filenames> | -l <filename> ] [-b <basedirectory>] [-c] [-o] [-N] [-p] [-h] 
            -f: comma-separated file names to process (the basedirectory (-b) is added if indicated)
            -b: specify the directory in which to find the specified files
            -t: provide files for testing (the basedirectory (-b) is added if indicated)
            -o: name of the pdf outfile of the decision tree
            -l: file that contains a list of filenames to load (overrides -f)
            -p: pad the previous statistics also
            -N: split the files provided in N to use for training and tot-N for testing (ignores -t)
            -c: cross validate on the individuals from the filenames provided with -f or -l (ignores -t, -N)
            -h: show this help and quit.
            ''')
            sys.exit()

    if specified_basedir:
        filenames = [basedir + f for f in filenames]
        test_filenames = [basedir + f for f in test_filenames]
    
    print('Parameters')
    print_parameters('\t', filenames=filenames, criterion=crit, min_samples_split=min_split, max_depth=max_depth, pad_prev=pad_prev, cross_validate_individuals=cross_validate_individuals)

    if cross_validate_individuals:
        #------------------------- Load datasets----------------------------------
        n_individuals = len(filenames) // 2
        folds = [load_cols([filenames[i],filenames[i+1]], pad_prev) for i in range(0, len(filenames), 2)]    # day and night
        train_accuracies = []
        test_accuracies = []
        for i in reversed(range(n_individuals)):
            (trainData, trainLabels),(testData, testLabels) = cross_validation_datasets(folds, i)
            #--------------------------Classify--------------------------------------
            classifier, train_results, test_results = decision_tree_classify(trainData, trainLabels,
                                                                            testData, testLabels,
                                                                            crit, min_split, max_depth)
            # Very Ad Hoc
            individual = filenames[2*i].split('/')[-1].split('_')[0]
            train_accuracies.append(train_results[0])
            test_accuracies.append(test_results[0])
            print('Test on individual %d (%s) train on the others' % (i,individual))
            print_results(train_results, test_results, labels)
        print('Mean train accuracies: %f' % np.mean(train_accuracies))
        print('Mean test accuracies: %f' % np.mean(test_accuracies))
    else:
        if test_provided:
            (trainData, trainLabels) = load_cols(filenames, pad_prev=pad_prev) #filenames[:1]
            (testData, testLabels) = load_cols(test_filenames, pad_prev=pad_prev) #filenames[1:]
        elif split:
            (trainData,trainLabels), (testData, testLabels) = load_cols_train_test(filenames, perc_train=0.7, pad_prev=pad_prev)
        elif N > -1:
            (trainData, trainLabels) = load_cols(filenames[:N], pad_prev=pad_prev)
            (testData, testLabels) = load_cols(filenames[N:], pad_prev=pad_prev)

        #--------------------------Classify--------------------------------------
        classifier, train_results, test_results = decision_tree_classify(trainData, trainLabels,
                                                                        testData, testLabels,
                                                                        crit, min_split, max_depth)
        print_results(train_results, test_results, labels)

        #----------------------Save tree image----------------------------------
        if pdffile != None:
            feature_names = features_names_pad(filenames[0], pad_prev) #csv_attributes(csv_filename)[:-1]
            print_pdf(classifier,pdffile, feature_names, labels)

if __name__ == '__main__':
    main()