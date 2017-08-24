from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from keras.models import Model
from utils import print_cm, reshape
from sklearn.model_selection import StratifiedKFold, train_test_split
from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence
import statistics as st
import numpy as np

def fitValidationSplit(model, X_train, y_train, split=2/7, epochs=1000, patience=10):
    model.fit(X_train, y_train, validation_split=split, verbose=2, epochs=epochs, callbacks=[EarlyStopping(monitor='loss', patience=100)]) #categorical_accuracy

def fitValidate(model, X_train, y_train, X_val, y_val, labels,  model_filename, class_weights, patience=10, resume=False, max_train_accuracy=0.99):
    best_accuracy = -1
    patience_count = 0
    i = 0
    if resume:
        model.load_weights(model_filename)
    while True:
        hist = model.fit(X_train, y_train, verbose=2, epochs=1, class_weight=class_weights)
        loss = hist.history['loss'][0]
        train_accuracy = hist.history['categorical_accuracy'][0]
        y_pred = model.predict_classes(X_val)

        # Compute scores on validation
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted')
        recall = recall_score(y_val, y_pred, average='weighted')
        f1 = f1_score(y_val, y_pred, average='weighted')

        report = classification_report(y_val, y_pred)
        print('\n\nReport at epoch '+str(i))
        print('Loss: ' + str(loss))
        print('Train accuracy: '+ str(train_accuracy))
        print()

        print('Validation accuracy: '+str(accuracy))
        print('Validation precision: '+str(precision))
        print('Validation recall: '+str(recall))
        print('Validation F1 Score: '+str(f1))
        print()
        print(report)

        cm = confusion_matrix(y_val, y_pred)
        print('Confusion matrix')
        print_cm(cm, labels)
        if best_accuracy == -1 or accuracy > best_accuracy:
            best_accuracy = accuracy
            #Save the model with the best accuracy on the validation set
            model.save_weights(model_filename)
            patience_count = 0
        else:
            patience_count += 1
            print('\nPatience count: '+str(patience_count)+'/'+str(patience))
            if patience_count == patience:
                print('Accuracy on validation stopped decreasing') #loss
                break
        i += 1
        if train_accuracy >= max_train_accuracy:
            print('Maximum train accuracy reached...')
            break
    model.load_weights(model_filename)

# Simplified method to insert into the slides
def train(model, X_train, y_train, X_val, y_val, model_filename, class_weights, patience=100):
    best_accuracy = -1
    patience_count = 0
    i = 0
    while True:
        # Train one epoch at a time
        model.fit(X_train, y_train, epochs=1, class_weight=class_weights)
        # Predict on validation
        y_pred = model.predict_classes(X_val)
        # Compute scores on validation
        accuracy = accuracy_score(y_val, y_pred)
        # Check the best accuracy
        if best_accuracy == -1 or accuracy > best_accuracy:
            best_accuracy = accuracy
            #Save the model with the best accuracy on the validation set
            model.save_weights(model_filename)
            patience_count = 0
        else:
            patience_count += 1
            # Early stopping with accuracy on validation with patience
            if patience_count == patience:
                print('Accuracy on validation stopped decreasing')
                break
        i += 1
    model.load_weights(model_filename)

def predict_test(model, X_test, y_test, labels):
    y_pred = model.predict_classes(X_test)
    report = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print()
    print('Test classification report')
    print('Accuracy: %f' % accuracy)
    print(report)
    print('Test confusion Matrix')
    cm = confusion_matrix(y_test, y_pred)
    print_cm(cm, labels)
    return accuracy_score(y_test, y_pred)

def pad_reshape(data, length, n_features, padding='pre'):
    ret_data = sequence.pad_sequences(data, maxlen=length, padding=padding, dtype='float', truncating='post', value=0.0)
    return reshape(ret_data, length, n_features)

#Currently it's not correct, the model should be reinitialized every time, to be tested...
def cross_validation_acc(model, X_data, y_data, labels, model_filename, class_weights, fn_x=lambda x:x, patience = 10, n_splits = 3, resume = False, train_size=5/7):
    start_model_config = model.get_config()
    skf = StratifiedKFold(n_splits = n_splits)
    best_accuracy = 0
    accuracies = []
    model_filename_best = 'cross_'+model_filename
    for train_index, test_index in skf.split(X_data, y_data):
        model = Model.from_config(start_model_config)
        X_data, y_data = np.array(X_data), np.array(y_data)
        X_train, X_test = X_data[train_index], X_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=train_size)
        X_train, X_valid, X_test = fn_x(X_train), fn_x(X_valid), fn_x(X_test)
        y_train = to_categorical(y_train, num_classes=len(labels))
        fitValidate(model, X_train, y_train, X_valid, y_valid, labels, model_filename, class_weights, patience, resume)
        accuracy = predict_test(model, X_test, y_test, labels)
        accuracies.append(accuracy)
        if accuracy > best_accuracy:
            model.save_weights(model_filename_best)
            best_accuracy = accuracy
    model.load_weights(model_filename_best)
    return st.mean(accuracies)