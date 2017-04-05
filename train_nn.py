from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU
from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
import numpy
from fetchdata import loadStratifiedDataset
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from utils import print_cm
import getopt
import sys

# fix random seed for reproducibility
numpy.random.seed(7)

#-------------------------------------Parameters------------------------------------------
sampling_period = 0.002
load_validation = False       #If loading a separate dataset or using a split for validation
seconds = 5
nLines = -1
aggregate = 1
filenames = ['../SleepEEG/rt 233_180511(1).txt','../SleepEEG/rt 233_180511(2).txt']
neurons = 125
transitions = False
n_features = 1
verbose = False
epochs = 1000
learning_rate = 0.00001

#----------------------------------Command line options-----------------------------------
try:
    opts, args = getopt.getopt(sys.argv[1:], 'f:n:a:s:e:ltvh')
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
    elif o == '-h':
        print('USAGE: python train_nn.py [-n <nLines>] [-f <filenames>] [-a <aggregate>] [-e <epochs>] [-s <seconds>] [-l] [-t] [-v] [-h]')
        sys.exit()
    else:
        sys.exit(3)

n_classes = 3 if not transitions else 4
length = int(seconds/(sampling_period*aggregate))
max_length = 2*length if transitions else length
labels = ['awake','nrem','rem'] if n_classes == 3 else ['nrem-awake','awake-nrem','nrem-rem', 'rem-aawake']

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
print('\tlearning rate: '+ str(learning_rate))
print()

#-----------------------------------load the dataset--------------------------------------
if load_validation:
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = loadStratifiedDataset(filenames, aggregate, nLines, seconds, sampling_period, load_validation, transitions=transitions, verbose=verbose)
else:
    (X_train, y_train), (X_test, y_test) = loadStratifiedDataset(filenames, aggr=aggregate, validation=load_validation, seconds=seconds, n=nLines, transitions=transitions, verbose=verbose)

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
model.add(GRU(neurons, input_shape=(max_length,1))) #for more layers ,return_sequences=True
#Add layers here
#model.add(LSTM(neurons))
model.add(Dense(n_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=learning_rate), metrics=['categorical_accuracy'])
print(model.summary())

#----------------------------------Fit and Validate---------------------------------------
def fitValidationSplit(model, X_train, y_train, split=2/7, epochs=1000):
    return model.fit(X_train, y_train, validation_split=split, verbose=2, epochs=epochs, callbacks=[EarlyStopping(monitor='loss', patience=100)]) #categorical_accuracy

def fitValidate(model, X_train, y_train, X_val, y_val, labels, epochs=1000, patience=100):
    prev_accuracy = -1
    patience_count = 0
    prev_loss = -1
    for i in range(epochs):
        hist = model.fit(X_train, y_train, verbose=2, epochs=1)
        loss = hist.history['loss'][0]
        print('Loss: ' + str(loss))
        y_pred = model.predict_classes(X_val)
        #accuracy = accuracy_score(y_val, y_pred)
        #print('Accuracy at epoch ' + str(i) + ': '+str(accuracy))
        '''if prev_accuracy == -1:
            prev_accuracy = accuracy
        elif prev_accuracy < accuracy:
            prev_accuracy = accuracy
            patience_count = 0
        else:
            patience_count += 1
            if patience_count == patience:
                print('Accuracy stopped increasing')
                break'''
        if prev_loss == -1:
            prev_loss = loss
        elif loss < prev_loss:
            prev_loss = loss
        else:
            patience_count += 1
            if patience_count == patience:
                print('Loss stopped decreasing')
                break
        report = classification_report(y_val, y_pred)
        print('Report at epoch '+str(i))
        print(report)
        cm = confusion_matrix(y_val, y_pred)
        print('Confusion matrix')
        print_cm(cm, labels)
    return hist

if load_validation:
    hist = fitValidate(model, X_train, y_train_cat, X_val, y_val, labels, epochs)
else:
    hist = fitValidationSplit(model,X_train, y_train_cat, epochs=epochs)

# use early stopping, or for on the epochs to stop for the best net. Measure on the validation on each epoch.
# measure accuracy every epoch, max accuracy. Save and load  weights.
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