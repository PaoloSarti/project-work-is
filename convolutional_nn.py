from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPool1D, BatchNormalization, Dropout, Flatten
from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
from fetchdata import loadStratifiedDataset, cachedDatalabels
from utils import reshape, class_weights_max, label_names, print_parameters
from training import fitValidate, predict_test, cross_validation_acc
import numpy as np

filenames = ['../SleepEEG/rt 233_180511(1).txt','../SleepEEG/rt 233_180511(2).txt', '../SleepEEG/rt 239_310511(1).txt', '../SleepEEG/rt 239_310511(2).txt' ]
nLines = -1
seconds = 10
n_classes = 3
aggregate = 5
sampling_period = 0.002
n_filters = 16 # 16, 32
kernel_size = 9 # Maybe... 3 5 7 9 ?
n_features = 1
dropout_rate = 0.2 # 0.5    (reduce overfitting)
patience = 100
model_filename = 'conv_weights.h5'
learning_rate = 0.00001
labels = label_names(n_classes)
cache = True
verbose = False
n_conv_layers = 3
transitions = True
crossvalidate = False
compare_individuals = True
pool_size = 2
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
                      compare_individuals=compare_individuals)

#------------------------------------Load the data----------------------------------------
if crossvalidate:
    X_data, y_data = cachedDatalabels(filenames, aggregate, nLines, seconds, sampling_period=sampling_period, transitions=transitions)
elif compare_individuals:
    (X_train, y_train), (X_val, y_val) = loadStratifiedDataset(filenames[2:],
                                                                aggregate,
                                                                nLines, 
                                                                seconds, 
                                                                sampling_period=sampling_period, 
                                                                validation=False, 
                                                                transitions=transitions, 
                                                                verbose=verbose,
                                                                cache=cache)
    (X_test, y_test) = cachedDatalabels(filenames[:2],
                                        aggregate,
                                        nLines,
                                        seconds,
                                        sampling_period=sampling_period,
                                        validation=False,
                                        transitions=transitions,
                                        verbose=verbose)
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
#model.add(Conv1D(n_filters, kernel_size, activation='relu', input_shape=(length,n_features)))

for i in range(n_conv_layers):
    if i == 0:
        model.add(Conv1D(n_filters, kernel_size, activation='relu', input_shape=(length,n_features)))
    else:
        model.add(Conv1D(n_filters, kernel_size, activation='relu'))
    model.add(MaxPool1D(pool_size))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
model.add(Flatten())
#model.add(Dense(neurons, activation='sigmoid'))
#model.add(Dropout(dropout_rate))
model.add(Dense(n_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=learning_rate), metrics=['categorical_accuracy'])
print(model.summary())

def pad_reshape(data, length, n_features):
    ret_data = sequence.pad_sequences(data, maxlen=length, padding='pre', dtype='float', truncating='post', value=0.0)
    return reshape(data, length, n_features)

if crossvalidate:
    #X_data = sequence.pad_sequences(X_data, maxlen=length, padding='pre', dtype='float', truncating='post', value=0.0)
    #X_data = reshape(X_data, length, n_features)
    acc = cross_validation_acc(model, X_data, y_data, labels, model_filename, class_weights, lambda x: pad_reshape(x, length, n_features), patience, n_splits = 5)
    print('Cross validated accuracy: '+str(acc))
else:
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
    fitValidate(model, X_train, y_train_cat, X_val, y_val, labels, model_filename, class_weights, patience)
    predict_test(model, X_test, y_test, labels)