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

filenames = ['../crunched_data/233_ff.csv', '../crunched_data/239_ff.csv']
learning_rate = 0.0001
patience = 1000
n_hidden_layers = 3
activation = 'relu'
resume = False
neurons = 20
pad_prev = True
neurons = 2 * neurons if pad_prev else neurons
compare_individuals = False
cols = ['SegmentStdDev', 'FreqAmplAvg', 'FreqAmplMaxFreq', 'SegmentLength', 'FreqAmplStdev', 'SegmentMax', 'SegmentMin', 'FreqAmplMin']

if compare_individuals:
    (trainData,trainLabels), (validData, validLabels) = load_cols_train_test(filenames[:1], pad_prev=pad_prev, cols=cols)
    (testData, testLabels) = load_cols(filenames[1:], pad_prev=pad_prev, cols=cols)
else:
    (trainData,trainLabels), (validData, validLabels), (testData, testLabels) = load_cols_train_valid_test(filenames, perc_train=0.5, perc_valid=0.2, pad_prev=pad_prev, cols=cols)

class_weights = class_weights_max(trainLabels)

print('Parameters')
print_parameters('\t', filenames=filenames, learning_rate=learning_rate, patience=patience, class_weights=class_weights, resume=resume, pad_prev=pad_prev, neurons=neurons)

#categorical
trainLabelsCat = to_categorical(trainLabels,num_classes=3)

#trainData = np.array(trainData)
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

fitValidate(model, normTrainData, trainLabelsCat, normValidData, validLabels, label_names(), 'simple_weights.h5',class_weights, patience, resume)

predict_test(model, normTestData, testLabels, label_names())