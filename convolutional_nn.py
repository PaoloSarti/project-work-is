# Script used to train and test convolutional neural networks

from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPool1D, BatchNormalization, Dropout, Flatten, Activation
from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop
from fetchdata import loadStratifiedDataset, cachedDatalabels, normalizeSegments, maxMinSegments
from utils import class_weights_max, label_names, print_parameters
from training import fitValidate, predict_test, cross_validation_acc, pad_reshape
import numpy as np
import functools

#-------------------------------------------Parameters---------------------------------------------
filenames = ['../SleepEEG/rt 233_180511(1).txt','../SleepEEG/rt 233_180511(2).txt', '../SleepEEG/rt 239_310511(1).txt', '../SleepEEG/rt 239_310511(2).txt' ]
nLines = -1                                 # How many lines to read from the csv file (-1: all of them)
seconds = 10                                # Number of seconds to consider at the start of the segment
n_classes = 3                              
aggregate = 5                               # Reduce the resolution of the signal by aggregating adjacent samples with the mean
sampling_period = 0.002
n_filters = 16                              # Convolutional Layer parameter (8,16,32)
kernel_size = 9                             # Size of the convolutional kernel (3 5 7 9...) 
n_features = 1                              # N features per sample (only the voltage in this case)
dropout_rate = 0.2                          # Parameter of the Droput layer to reduce overfitting (0.2-0.5)
patience = 100                              # N of iterations without improving validation accuracy to wait
model_filename = 'conv_weights.h5'          # Weights filename to store the best network (best accuracy on validation)
learning_rate = 0.00001
labels = label_names(n_classes)             # Names of the classes
cache = True                                # Cache on the filesystem the prepared data (improve drastically the performance for subsequent runs with the same parameters)
verbose = True                             
n_conv_layers = 3                           # How many hidden convolutional layers (Conv, MaxPool, BatchNorm, Dropout) to use
transitions = True                          # Consider also the same number of seconds before the start of the sequence
crossvalidate = False                       # Cross validation on the entire dataset (mutually exclusive with compare_individuals). Currently not correct...
compare_individuals = False                 # Train the model on one individual, test on the other
pool_size = 2                               # Size of the MaxPool layer
#l = int(seconds/(sampling_period*aggregate))
#length = l if not transitions else 2 * l

print('Parameters:')
print_parameters('\t',filenames=filenames,
                      nLines=nLines,
                      seconds=seconds,
                      aggregate=aggregate,
                      sampling_period=sampling_period,
                      n_filters=n_filters,
                      kernel_size=kernel_size,
                      pool_size=pool_size,
                      n_features=n_features,
                      dropout_rate=dropout_rate,
                      patience=patience,
                      model_filename=model_filename,
                      learning_rate=learning_rate,
                      labels=labels,
                      cache=cache,
                      verbose=verbose,
                      transitions=transitions,
                      compare_individuals=compare_individuals,
                      crossvalidate=crossvalidate)

#------------------------------------Load the data----------------------------------------
if crossvalidate:
    X_data, y_data = cachedDatalabels(filenames, aggregate, nLines, seconds, sampling_period=sampling_period, transitions=transitions)
elif compare_individuals:
    (X_train, y_train), (X_val, y_val) = loadStratifiedDataset(filenames[:2],
                                                                aggregate,
                                                                nLines, 
                                                                seconds, 
                                                                sampling_period=sampling_period, 
                                                                validation=False, 
                                                                transitions=transitions, 
                                                                verbose=verbose,
                                                                cache=cache)
    (X_test, y_test) = cachedDatalabels(filenames[2:],
                                        aggregate,
                                        nLines,
                                        seconds,
                                        sampling_period=sampling_period,
                                        validation=False,
                                        transitions=transitions,
                                        verbose=verbose)
    X_test = normalizeSegments(X_test)
else:
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = loadStratifiedDataset(filenames,
                                                                                aggregate,
                                                                                nLines,
                                                                                seconds,
                                                                                sampling_period=sampling_period,
                                                                                verbose=verbose,
                                                                                cache=cache,
                                                                                transitions=transitions)
if crossvalidate:
    length = len(X_data[0])
    class_weights = class_weights_max(y_data)
else:
    length = len(X_train[0])
    class_weights = class_weights_max(y_train)

print_parameters('\t', length=length, class_weights=class_weights)

#---------------------------------------Model---------------------------------------------
model = Sequential()
for i in range(n_conv_layers):
    if i == 0:
        model.add(Conv1D(n_filters, kernel_size, input_shape=(length,n_features), strides=1))
    else:
        model.add(Conv1D(n_filters, kernel_size))
    model.add(MaxPool1D(pool_size))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))
model.add(Flatten())
model.add(Dense(n_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=learning_rate), metrics=['categorical_accuracy'])
print(model.summary())

prepare_data = functools.partial(pad_reshape, length=length, n_features=n_features, padding='post')

if crossvalidate:
    acc = cross_validation_acc(model, X_data, y_data, labels, model_filename, class_weights, prepare_data, patience, n_splits = 5)
    print('Cross validated accuracy: '+str(acc))
else:
    #-----------------------------------categorical-------------------------------------------
    y_train_cat = to_categorical(y_train, num_classes=n_classes)
    #-----------------------------------prepare data------------------------------------------
    X_train = prepare_data(X_train)
    X_val = prepare_data(X_val)
    X_test = prepare_data(X_test)
    fitValidate(model, X_train, y_train_cat, X_val, y_val, labels, model_filename, class_weights, patience)
    predict_test(model, X_test, y_test, labels)