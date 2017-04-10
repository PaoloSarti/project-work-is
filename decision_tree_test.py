from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from fetchdata import load_segment_statistics_train_test, load_segment_statistics, load_cols_train_test
from utils import print_cm, print_parameters
import pydotplus

filenames = ['../crunched_data/233_f.csv','../crunched_data/239_f.csv']
pdffile = '233_239.pdf'
dotfile = '239_tree.dot'
labels = ['Awake','Nrem','Rem']
crit = 'gini'
min_split = 20
max_depth = 5

print('Parameters')
print_parameters('\t', filenames=filenames, criterion=crit, min_samples_split=min_split, max_depth=max_depth)

(trainData,trainLabels), (testData, testLabels) = load_cols_train_test(filenames, perc_train=0.8) #load_segment_statistics_train_test(filenames, perc_train=0.8)

classifier = tree.DecisionTreeClassifier(criterion='gini',#entrpy
                                         min_samples_split=20,
                                         max_depth=5)

classifier.fit(trainData, trainLabels)

trainPred = classifier.predict(trainData)
predicted = classifier.predict(testData)

print('On training set')
print(classification_report(trainLabels, trainPred))
print('Confusion matrix')
cm = confusion_matrix(trainLabels,trainPred)
print_cm(cm, labels)
print()
print('On test set')
print(classification_report(testLabels,predicted))
print('Confusion matrix')
cm = confusion_matrix(testLabels,predicted)
print_cm(cm, labels)

def print_pdf(classifier, filename):
    dot_data = tree.export_graphviz(classifier,
                                    out_file=None,
                                    feature_names=['Len','Min','Max','Avg','StdDev'],
                                    class_names=['Awake','NREM','REM']) 
    graph = pydotplus.graph_from_dot_data(dot_data) 
    graph.write_pdf(filename)

#print('Cross validation')
#data,labels = load_segment_statistics(filenames)
#cross_val_score(classifier,data, labels,cv=8, scoring=)

#print_pdf(classifier,pdffile)
#tree.export_graphviz(classifier,out_file=dotfile)