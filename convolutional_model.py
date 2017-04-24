from keras.models import Sequential
from keras.layers import Conv1D, MaxPool1D, BatchNormalization, Dropout, Dense
from keras.optimizers import RMSprop

def convolutional_model(n_conv_layers, kernel_size, input_shape, dropout_rate, n_classes, learning_rate):
    model = Sequential()
    #model.add(Conv1D(n_filters, kernel_size, activation='relu', input_shape=(length,n_features)))
    for i in range(n_conv_layers):
        if i == 0:
            model.add(Conv1D(n_filters, kernel_size, activation='relu', input_shape=(length,n_features)))
        else:
            model.add(Conv1D(n_filters, kernel_size, activation='relu'))
        model.add(MaxPool1D())
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
    model.add(Flatten())
    #model.add(Dense(neurons, activation='sigmoid'))
    #model.add(Dropout(dropout_rate))
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=learning_rate), metrics=['categorical_accuracy'])
    print(model.summary())