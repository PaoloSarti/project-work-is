from keras.layers import Dense,Activation
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from fetchdata import load_segment_statistics_train_test, load_segment_statistics_train_valid_test, load_cols_train_valid_test
from utils import normalizeColumns, print_cm, label_names, print_parameters, class_weights_max
from training import fitValidate, predict_test
import numpy as np
import sys

filenames = ['../crunched_data/239_f.csv','../crunched_data/233_f.csv']
learning_rate = 0.0001
patience = 100
n_hidden_layers = 1
activation = 'sigmoid'
resume = False
neurons = 20
pad_prev = True

#(trainData,trainLabels), (testData, testLabels) = load_segment_statistics_train_test(filenames, perc_train=0.8)
#load_segment_statistics_train_valid_test(filenames, perc_train=0.7, perc_valid=0.1)
(trainData,trainLabels), (validData, validLabels), (testData, testLabels) = load_cols_train_valid_test(filenames, perc_train=0.7, perc_valid=0.1, pad_prev=pad_prev)

class_weights = class_weights_max(trainLabels)

print('Parameters')
print_parameters('\t', filenames=filenames, learning_rate=learning_rate, patience=patience, class_weights=class_weights, resume=resume, pad_prev=pad_prev)

#categorical
trainLabelsCat = to_categorical(trainLabels,num_classes=3)

#trainData = np.array(trainData)
#Normalized
normTrainData = normalizeColumns(trainData)
normValidData = normalizeColumns(validData)
normTestData = normalizeColumns(testData)

model = Sequential()
model.add(Dense(neurons, input_dim=normTrainData.shape[1], activation=activation))
model.add(BatchNormalization())
for i in range(n_hidden_layers-1):
    model.add(Dense(neurons, activation=activation))
    model.add(BatchNormalization())
model.add(Dense(3, activation='softmax'))

model.compile(optimizer=RMSprop(lr=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())

fitValidate(model, normTrainData, trainLabelsCat, normValidData, validLabels, label_names(), 'simple_weights.h5',class_weights, patience, resume)

predict_test(model, normTestData, testLabels, label_names())