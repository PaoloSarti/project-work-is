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

#---------------------------Parameters-------------------------------------
files = ['233_day.csv', '239_day.csv', '243_day.csv', '258_day.csv', '259_day.csv', '268_day.csv', '279_day.csv', '305_day.csv', '334_day.csv', '344_day.csv']
#'233_night.csv', '239_night.csv', '243_night.csv', '258_night.csv', '259_night.csv', '268_night.csv', '279_night.csv', '305_night.csv', '334_night.csv', '344_night.csv']
basedir = '../crunched_data/normalized/'
filenames = [basedir + f for f in files]
learning_rate = 0.0001
patience = 100
n_hidden_layers = 3
activation = 'relu'
resume = False
neurons = 20
pad_prev = True
neurons = 2 * neurons if pad_prev else neurons #double the neuron count if the inputs are doubled
compare_individuals = True
cols = None #['SegmentStdDev', 'FreqAmplAvg', 'FreqAmplMaxFreq', 'SegmentLength', 'FreqAmplStdev', 'SegmentMax', 'SegmentMin', 'FreqAmplMin']

#------------------------- Load datasets----------------------------------
if compare_individuals:
    (trainData,trainLabels), (validData, validLabels) = load_cols_train_test(filenames[:-1], pad_prev=pad_prev, cols=cols)
    (testData, testLabels) = load_cols(filenames[-1:], pad_prev=pad_prev, cols=cols)
else:
    (trainData,trainLabels), (validData, validLabels), (testData, testLabels) = load_cols_train_valid_test(filenames, perc_train=0.5, perc_valid=0.2, pad_prev=pad_prev, cols=cols)

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