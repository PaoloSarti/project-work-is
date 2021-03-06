# Script used to train and test lstm neural networks

from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU
from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
import numpy
from fetchdata import loadStratifiedDataset
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from utils import print_cm, label_names, class_weights_max, print_parameters, reshape
from training import fitValidate, fitValidationSplit, predict_test
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
filenames = ['../SleepEEG/rt 233_180511(1).txt','../SleepEEG/rt 233_180511(2).txt', '../SleepEEG/rt 239_310511(1).txt', '../SleepEEG/rt 239_310511(2).txt' ]
neurons = 125
transitions = False
n_features = 1
verbose = False
epochs = 1000
patience = 10
learning_rate = 0.00001
cache = False
resume = False
 
#----------------------------------Command line options-----------------------------------
try:
    opts, args = getopt.getopt(sys.argv[1:], 'f:n:a:s:e:p:lrtvch')
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
    elif o == '-r':
        resume = True
    elif o == '-h':
        print('''USAGE: python train_nn.py [-n <nLines>] [-f <filenames>] [-a <aggregate>] [-e <epochs>] [-s <seconds>] [-p <patience>] [-l] [-t] [-v] [-c] [-r] [-h] 
        -n <nLines>: number of lines to fetch from the files. Default: all.
        -a <aggregate>: number of eeg values to aggregate together, taking the average. Default: 5.
        -e <epochs>: number of max epochs. Default 1000.
        -s <seconds>: how many seconds of eeg to consider per segment. Default: 5.
        -p <patience>: how many epochs without improvement of the minimal loss to tolerate before stopping early. Default: 10.
        -l: load a separate dataset for validation, otherwise, a split in the training set is used. More information can be given during training in this way.
        -t: consider s seconds of eeg before the transition. The number of classes takes into account the transitions.
        -c: cache the loaded segments into a file, for a much faster loading the next time (with the same parameters).
        -r: resume: load the weights from the previous computation
        -v: verbose. Log info about the segments being loaded.
        -h: show this help and quit.
        ''')
        sys.exit()
    else:
        sys.exit(3)

n_classes = 3 if not transitions else 4
length = int(seconds/(sampling_period*aggregate))
max_length = 2*length if transitions else length
labels = label_names(n_classes)
model_filename = 'weights.h5'

#-----------------------------------load the dataset--------------------------------------
if load_validation:
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = loadStratifiedDataset(filenames, aggregate, nLines, seconds, sampling_period, load_validation, transitions=transitions, verbose=verbose,cache=cache)
else:
    (X_train, y_train), (X_test, y_test) = loadStratifiedDataset(filenames, aggr=aggregate, validation=load_validation, seconds=seconds, n=nLines, transitions=transitions, verbose=verbose, cache=cache)

class_weights = class_weights_max(y_train)

#----------------------------------print parameters----------------------------------------
print('Parameters:')
print_parameters('\t', sampling_period=sampling_period,
                       load_validation=load_validation,
                       seconds=seconds, nLines=nLines,
                       aggregate=aggregate,
                       filenames=filenames,
                       neurons=neurons,
                       transitions=transitions,
                       n_features=n_features,
                       n_classes=n_classes,
                       max_length=max_length,
                       epochs=epochs,
                       patience=patience,
                       learning_rate=learning_rate,
                       class_weights=class_weights,
                       resume=resume)
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

#---------------------------------------Reshape-------------------------------------------
X_train = reshape(X_train, max_length, n_features)
if load_validation:
    X_val = reshape(X_val, max_length, n_features)
X_test = reshape(X_test, max_length, n_features)

#---------------------------------------Model---------------------------------------------
model = Sequential()
model.add(LSTM(neurons, input_shape=(max_length,n_features))) #for more layers ,return_sequences=True
model.add(Dense(n_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=learning_rate), metrics=['categorical_accuracy'])
print(model.summary())

#----------------------------------Fit and Validate---------------------------------------

if load_validation:
    fitValidate(model, X_train, y_train_cat, X_val, y_val, labels, model_filename, class_weights, patience, resume)
else:
    fitValidationSplit(model, X_train, y_train_cat, split=2/7, epochs=epochs, patience=patience)

predict_test(model, X_test, y_test, labels)