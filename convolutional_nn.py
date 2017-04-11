from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPool1D, BatchNormalization, Dropout, Flatten
from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
from fetchdata import loadStratifiedDataset
from utils import reshape, class_weights_max, label_names, print_parameters
from training import fitValidate, predict_test
import numpy as np


filenames = ['../SleepEEG/rt 233_180511(1).txt','../SleepEEG/rt 233_180511(2).txt', '../SleepEEG/rt 239_310511(1).txt', '../SleepEEG/rt 239_310511(2).txt' ]
nLines = -1
seconds = 5
n_classes = 3
aggregate = 5
sampling_period = 0.002
n_filters = 8 # 16, 32
kernel_size = 7 # Maybe.... 3 5 ?
n_features = 1
dropout_rate = 0.2 # 0.5    (reduce overfitting)
length = int(seconds/(sampling_period*aggregate))
patience = 100
model_filename = 'conv_weights.h5'
learning_rate = 0.00001
labels = label_names(n_classes)
cache = True
verbose = False

print('Parameters:')
print_parameters('\t',filenames=filenames,
                      nLines=nLines,
                      seconds=seconds,
                      aggregate=aggregate,
                      sampling_period=sampling_period,
                      n_filters=n_filters,
                      kernel_size=kernel_size,
                      n_features=n_features,
                      dropout_rate=dropout_rate,
                      length=length,
                      patience=patience,
                      model_filename=model_filename,
                      learning_rate=learning_rate,
                      labels=labels,
                      cache=cache,
                      verbose=verbose)

#------------------------------------Load the data----------------------------------------
(X_train, y_train), (X_val, y_val), (X_test, y_test) = loadStratifiedDataset(filenames,
                                                                             aggregate,
                                                                             nLines,
                                                                             seconds,
                                                                             sampling_period=sampling_period,
                                                                             verbose=verbose,
                                                                             cache=cache)
class_weights = class_weights_max(y_train)
#-----------------------------------categorical-------------------------------------------
y_train_cat = to_categorical(y_train, num_classes=n_classes)

#--------------------------------------truncate-------------------------------------------
X_train = sequence.pad_sequences(X_train, maxlen=length, padding='pre', dtype='float',truncating='post',value=0.0)
X_val = sequence.pad_sequences(X_val, maxlen=length, padding='pre', dtype='float',truncating='post',value=0.0)
X_test = sequence.pad_sequences(X_test, maxlen=length, padding='pre', dtype='float',truncating='post',value=0.0)

#--------------------------------------reshape--------------------------------------------
X_train = reshape(X_train, length, n_features)
X_val = reshape(X_val, length, n_features)
X_test = reshape(X_test, length, n_features)
print('X_train shape: '+str(X_train.shape))
#---------------------------------------Model---------------------------------------------
model = Sequential()
model.add(Conv1D(n_filters, kernel_size, activation='relu', input_shape=(length,n_features)))
model.add(MaxPool1D())
model.add(BatchNormalization())
model.add(Dropout(dropout_rate))

model.add(Conv1D(n_filters, kernel_size, activation='relu'))
model.add(MaxPool1D())
model.add(BatchNormalization())
model.add(Dropout(dropout_rate))

model.add(Flatten())
model.add(Dense(250, activation='sigmoid'))
model.add(Dense(n_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=learning_rate), metrics=['categorical_accuracy'])
print(model.summary())

fitValidate(model, X_train, y_train_cat, X_val, y_val, labels, model_filename, class_weights, patience)

predict_test(model, X_test, y_test, labels)