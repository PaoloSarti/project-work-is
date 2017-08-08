from keras.layers import Dense,Activation
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from fetchdata import load_cols_train_valid_test, load_cols, load_cols_train_test
from utils import normalizeColumns, print_cm, label_names, print_parameters, class_weights_max
from training import fitValidate, predict_test
import numpy as np
import sys
import getopt

#---------------------------Parameters-------------------------------------
filenames = ['233_day.csv', '239_day.csv', '243_day.csv', '258_day.csv', '259_day.csv', '268_day.csv', '279_day.csv', '305_day.csv', '334_day.csv', '344_day.csv',
'233_night.csv', '239_night.csv', '243_night.csv', '258_night.csv', '259_night.csv', '268_night.csv', '279_night.csv', '305_night.csv', '334_night.csv', '344_night.csv']
basedir = '../crunched_data/standardized/'
specified_basedir = False 
test_provided = False
test_filenames = []
learning_rate = 0.0001
patience = 100
n_hidden_layers = 3
activation = 'relu'
resume = False
neurons = 20
pad_prev = True
neurons = 2 * neurons if pad_prev else neurons #double the neuron count if the inputs are doubled
compare_individuals = False
cols = None #['SegmentStdDev', 'FreqAmplAvg', 'FreqAmplMaxFreq', 'SegmentLength', 'FreqAmplStdev', 'SegmentMax', 'SegmentMin', 'FreqAmplMin']
cross_validate_individuals = False
N = -1
split = True

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
        print('''USAGE: python simple_nn.py [-f <filenames>] [-t <filenames> | -l <filename> ] [-b <basedirectory>] [-c] [-N] [-p] [-h] 
        -f: comma-separated file names to process (the basedirectory (-b) is added if indicated)
        -b: specify the directory in which to find the specified files
        -t: provide files for testing (the basedirectory (-b) is added if indicated)
        -l: file that contains a list of filenames to load (overrides -f)
        -p: pad the previous statistics also
        -c: cross validate on the individuals from the filenames provided with -f or -l (ignores -t, -N)
        -h: show this help and quit.
        ''')
        sys.exit()

if specified_basedir:
    filenames = [basedir + f for f in filenames]
    test_filenames = [basedir + f for f in test_filenames]

#------------------------- Load datasets----------------------------------
if compare_individuals:
    (trainData,trainLabels), (validData, validLabels) = load_cols_train_test(filenames[:-1], pad_prev=pad_prev, cols=cols)
    (testData, testLabels) = load_cols(filenames[-1:], pad_prev=pad_prev, cols=cols)
elif split:
    (trainData,trainLabels), (validData, validLabels), (testData, testLabels) = load_cols_train_valid_test(filenames, perc_train=0.5, perc_valid=0.2, pad_prev=pad_prev, cols=cols)
elif N!=-1:
    (trainData,trainLabels), (validData, validLabels) = load_cols_train_test(filenames[:N], pad_prev=pad_prev, cols=cols)
    (testData, testLabels) = load_cols(filenames[N:], pad_prev=pad_prev, cols=cols)

class_weights = class_weights_max(trainLabels)

print('Parameters')
print_parameters('\t', filenames=filenames, learning_rate=learning_rate, patience=patience, class_weights=class_weights, resume=resume, pad_prev=pad_prev, neurons=neurons)

#--------------------------Prepare dataset-------------------------------
#categorical
trainLabelsCat = to_categorical(trainLabels,num_classes=3)

#Normalized
normTrainData = normalizeColumns(trainData)
normValidData = normalizeColumns(validData)
normTestData = normalizeColumns(testData)

def simple_model():
    model = Sequential()
    model.add(Dense(neurons, input_dim=normTrainData.shape[1], activation='sigmoid'))
    model.add(Dense(3, activation='softmax'))
    return model

def deep_model():
    model=Sequential()
    model.add(Dense(neurons, input_dim=normTrainData.shape[1]))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    for i in range(n_hidden_layers-1):
        model.add(Dense(neurons))
        model.add(BatchNormalization())
        model.add(Activation(activation))
    return model

#------------------------Model------------------------------------------
model = Sequential()
model.add(Dense(neurons, input_dim=normTrainData.shape[1]))
if activation == 'relu':
    model.add(BatchNormalization())
model.add(Activation(activation))
for i in range(n_hidden_layers-1):
    model.add(Dense(neurons))
    if activation == 'relu':
        model.add(BatchNormalization())
    model.add(Activation(activation))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer=RMSprop(lr=learning_rate),
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])
print(model.summary())

#-------------------------Train and Test---------------------------------
fitValidate(model, normTrainData, trainLabelsCat, normValidData, validLabels, label_names(), 'simple_weights.h5',class_weights, patience, resume)

predict_test(model, normTestData, testLabels, label_names())