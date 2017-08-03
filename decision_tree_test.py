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

def main():
    #---------------------------Parameters-------------------------------------
    basedir = '../crunched_data/'
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

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'f:o:b:pl:t:h')
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
        elif o == '-h':
            print('''USAGE: python decision_tree_test.py [-f <filenames>] [-t <filenames>] [-b <basedirectory>] [-l <filename>] [-o] [-p] [-h] 
            -f: comma-separated file names to process
            -b: specify the directory in which to find the files
            -t: provide files for testing
            -o: name of the pdf outfile of the decision tree
            -l: file that contains a list of filenames to load (overrides -f)
            -p: pad the previous statistics also
            -h: show this help and quit.
            ''')
            sys.exit()

    if specified_basedir:
        filenames = [basedir + f for f in filenames]
        test_filenames = [basedir + f for f in test_filenames]
    
    print('Parameters')
    print_parameters('\t', filenames=filenames, criterion=crit, min_samples_split=min_split, max_depth=max_depth)

    #------------------------- Load datasets----------------------------------
    if test_provided:
        (trainData, trainLabels) = load_cols(filenames, pad_prev=pad_prev) #filenames[:1]
        (testData, testLabels) = load_cols(test_filenames, pad_prev=pad_prev) #filenames[1:]
    else:
        (trainData,trainLabels), (testData, testLabels) = load_cols_train_test(filenames, perc_train=0.7, pad_prev=pad_prev) #load_segment_statistics_train_test(filenames, perc_train=0.8)

    #---------------------Decision Tree Classifier----------------------------
    classifier = tree.DecisionTreeClassifier(criterion=crit,min_samples_split=min_split,max_depth=max_depth)
    classifier.fit(trainData, trainLabels)
    trainPred = classifier.predict(trainData)
    predicted = classifier.predict(testData)

    #------------------------Print results------------------------------------
    print('On training set')
    train_acc = accuracy_score(trainLabels, trainPred)
    print('Accuracy: %f' % train_acc)
    print(classification_report(trainLabels, trainPred))
    print('Confusion matrix')
    cm = confusion_matrix(trainLabels,trainPred)
    print_cm(cm, labels)
    print()
    print('On test set')
    test_acc = accuracy_score(testLabels, predicted)
    print('Accuracy: %f' % test_acc)
    print(classification_report(testLabels,predicted))
    print('Confusion matrix')
    cm = confusion_matrix(testLabels,predicted)
    print_cm(cm, labels)

    #----------------------Save tree image----------------------------------
    def features_names_pad(csv_filename, pad_prev):
        prev_cur = ['Prev', 'Cur']
        names = csv_attributes(csv_filename)[:-1]   #skip the class
        if not pad_prev:
            return names
        else:
            return [e[1]+str(prev_cur[e[0] // len(names)]) for e in enumerate(names + names)]

    def print_pdf(classifier, filename, csv_filename, pad_prev):
        'Print the decision tree on pdf'
        dot_data = tree.export_graphviz(classifier,
                                        out_file=None,
                                        feature_names=  features_names_pad(csv_filename, pad_prev),  #csv_attributes(csv_filename)[:-1]
                                        class_names=labels) 
        graph = pydotplus.graph_from_dot_data(dot_data) 
        graph.write_pdf(filename)

    if pdffile != None:
        print_pdf(classifier,pdffile, filenames[0], pad_prev)

if __name__ == '__main__':
    main()