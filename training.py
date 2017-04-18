from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utils import print_cm

def fitValidationSplit(model, X_train, y_train, split=2/7, epochs=1000, patience=10):
    model.fit(X_train, y_train, validation_split=split, verbose=2, epochs=epochs, callbacks=[EarlyStopping(monitor='loss', patience=100)]) #categorical_accuracy

def fitValidate(model, X_train, y_train, X_val, y_val, labels,  model_filename, class_weights, patience=10, resume=False):
    prev_accuracy = -1
    patience_count = 0
    prev_loss = -1
    i = 0
    if resume:
        model.load_weights(model_filename)
    while True:
        hist = model.fit(X_train, y_train, verbose=2, epochs=1, class_weight=class_weights)
        loss = hist.history['loss'][0]
        train_accuracy = hist.history['categorical_accuracy'][0]
        y_pred = model.predict_classes(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        report = classification_report(y_val, y_pred)
        print('\n\nReport at epoch '+str(i))
        print('Loss: ' + str(loss))
        print('Train accuracy: '+ str(train_accuracy))
        print('Validation accuracy: '+str(accuracy))
        print(report)
        cm = confusion_matrix(y_val, y_pred)
        print('Confusion matrix')
        print_cm(cm, labels)
        if prev_accuracy == -1 or accuracy > prev_accuracy:
            prev_accuracy = accuracy
            #Save the model with the best accuracy on the validation set
            model.save_weights(model_filename)
            patience_count = 0
        #if prev_loss == -1:
        #    prev_loss = loss
        #elif loss < prev_loss:
        #    prev_loss = loss
        #    patience_count = 0
        else:
            patience_count += 1
            print('\nPatience count: '+str(patience_count)+'/'+str(patience))
            if patience_count == patience:
                print('Accuracy on validation stopped decreasing') #loss
                break
        i += 1
    model.load_weights(model_filename)

def predict_test(model, X_test, y_test, labels):
    y_pred = model.predict_classes(X_test)
    report = classification_report(y_test, y_pred)
    print()
    print('Test classification report')
    print(report)
    print('Test confusion Matrix')
    cm = confusion_matrix(y_test, y_pred)
    print_cm(cm, labels)