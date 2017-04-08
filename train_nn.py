from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU
from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
import numpy
from fetchdata import loadStratifiedDataset
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from utils import print_cm, labels, class_weight_count
from training import fitValidate, fitValidationSplit
import getopt
import sys

# fix random seed for reproducibility
numpy.random.seed(7)

#-------------------------------------Parameters------------------------------------------
sampling_period = 0.002
load_validation = False       #Load a separate dataset or use a split for validation
seconds = 5
nLines = -1
aggregate = 5
filenames = ['../SleepEEG/rt 233_180511(1).txt','../SleepEEG/rt 233_180511(2).txt', '../SleepEEG/rt 239_310511(1).txt', 'rt 239_310511(2).txt' ]
neurons = 125
transitions = False
n_features = 1
verbose = False
epochs = 1000
patience = 10
learning_rate = 0.00001
cache = False
 
#----------------------------------Command line options-----------------------------------
try:
    opts, args = getopt.getopt(sys.argv[1:], 'f:n:a:s:e:p:ltvch')
except getopt.GetoptError as err:
    print(err)
    sys.exit(2)
for o, a in opts:
    if o == '-f':
        filenames = a.split(',')
    elif o == '-n':
        nLines = int(a)
    elif o == '-a':
        aggregate = int(a)
    elif o == '-s':
        seconds = int(a)
    elif o == '-t':
        transitions = True
    elif o == '-l':
        load_validation = True
    elif o == '-v':
        verbose = True
    elif o == '-e':
        epochs = int(a)
    elif o == '-c':
        cache = True
    elif o == '-p':
        patience = int(a)
    elif o == '-h':
        print('''USAGE: python train_nn.py [-n <nLines>] [-f <filenames>] [-a <aggregate>] [-e <epochs>] [-s <seconds>] [-p <patience>] [-l] [-t] [-v] [-c] [-h]
        -n <nLines>: number of lines to fetch from the files. Default: all.
        -a <aggregate>: number of eeg values to aggregate together, taking the average. Default: 5.
        -e <epochs>: number of max epochs. Default 1000.
        -s <seconds>: how many seconds of eeg to consider per segment. Default: 5.
        -p <patience>: how many epochs without improvement of the minimal loss to tolerate before stopping early. Default: 10.
        -l: load a separate dataset for validation, otherwise, a split in the training set is used. More information can be given during training in this way.
        -t: consider s seconds of eeg before the transition. The number of classes takes into account the transitions.
        -c: cache the loaded segments into a file, for a much faster loading the next time (with the same parameters).
        -h: show this help and quit.
        ''')
        sys.exit()
    else:
        sys.exit(3)

n_classes = 3 if not transitions else 4
length = int(seconds/(sampling_period*aggregate))
max_length = 2*length if transitions else length
labels = labels(n_classes)
model_filename = 'weights.h5'

#-----------------------------------load the dataset--------------------------------------
if load_validation:
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = loadStratifiedDataset(filenames, aggregate, nLines, seconds, sampling_period, load_validation, transitions=transitions, verbose=verbose,cache=cache)
else:
    (X_train, y_train), (X_test, y_test) = loadStratifiedDataset(filenames, aggr=aggregate, validation=load_validation, seconds=seconds, n=nLines, transitions=transitions, verbose=verbose, cache=cache)

class_weights = class_weight_count(y_train)

#----------------------------------print parameters----------------------------------------
print('Parameters:')
print('\tsampling_period: '+str(sampling_period))
print('\tload_validation: '+str(load_validation))
print('\tseconds: '+str(seconds)+ ' (-1 is no limit)')
print('\tnLines: '+str(nLines)+ ' (-1 is all of them)')
print('\taggregate: '+ str(aggregate))
print('\tfilenames: '+ str(filenames))
print('\tneurons: '+ str(neurons))
print('\ttransitions: '+ str(transitions))
print('\tn_features (per timestamp): '+ str(n_features))
print('\tn_classes: '+ str(n_classes))
print('\tmax_length of the sequence: '+ str(max_length))
print('\tmax epochs: '+ str(epochs))
print('\tpatience: '+ str(patience))
print('\tlearning rate: '+ str(learning_rate))
print('\tclass_weights: '+str(class_weights))
print()

#------------------------------------to categorical----------------------------------------
y_train_cat = to_categorical(y_train, num_classes=n_classes)
y_test_cat = to_categorical(y_test, num_classes=n_classes)

#---------------------------Truncate and pad input sequences-------------------------------
print('Pad sequences. Max length: ' + str(max_length))
X_train = sequence.pad_sequences(X_train, maxlen=max_length, padding='pre', dtype='float',truncating='post',value=0.0)
if load_validation:
    X_val = sequence.pad_sequences(X_val, maxlen=max_length, padding='pre', dtype='float',truncating='post',value=0.0)
X_test = sequence.pad_sequences(X_test, maxlen=max_length, padding='pre', dtype='float',truncating='post',value=0.0)

def reshape(dataset, seq_len, n_features):
    l = dataset.shape[0]
    return dataset.reshape((l,seq_len,n_features))

#---------------------------------------Reshape-------------------------------------------
X_train = reshape(X_train, max_length, n_features)
if load_validation:
    X_val = reshape(X_val, max_length, n_features)
X_test = reshape(X_test, max_length, n_features)

#---------------------------------------Model---------------------------------------------
model = Sequential()
model.add(LSTM(neurons, input_shape=(max_length,1))) #for more layers ,return_sequences=True
#Add layers here
#model.add(LSTM(neurons))
model.add(Dense(n_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=learning_rate), metrics=['categorical_accuracy'])
print(model.summary())

#----------------------------------Fit and Validate---------------------------------------

if load_validation:
    fitValidate(model, X_train, y_train_cat, X_val, y_val, labels, model_filename, class_weights, patience)
else:
    fitValidationSplit(model, X_train, y_train_cat, split=2/7, epochs=epochs, patience=patience)

# accuracy up, loss down, at least on train.

# Predict, do the confusion matrix. Precision recall and f1, scikit-learn
# Then we assess the performance of the model with the test
scores = model.evaluate(X_test, y_test_cat, verbose=1)
#print(str(model.metrics_names))
#print(str(scores))
print("Accuracy: %.2f%%" % (scores[1]*100))

y_pred = model.predict_classes(X_test)
report = classification_report(y_test, y_pred)
print('Report test')
print(report)
print('Test confusion Matrix')
cm = confusion_matrix(y_test, y_pred)
print_cm(cm, labels)