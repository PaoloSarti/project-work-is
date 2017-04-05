from keras.layers import Dense,Activation
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from fetchdata import load_segment_statistics_train_test
from utils import normalizeColumns, print_cm
import numpy as np

filenames = ['../crunched_data/239.csv','../crunched_data/233.csv']

(trainData,trainLabels), (testData, testLabels) = load_segment_statistics_train_test(filenames, perc_train=0.8)

#categorical
trainLabelsCat = to_categorical(trainLabels,num_classes=3)

model = Sequential()
model.add(Dense(10, input_dim=5, activation='sigmoid'))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer=RMSprop(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())

trainData = np.array(trainData)

normTrainData = normalizeColumns(trainData)
normTestData = normalizeColumns(testData)
#trainLabelsNp = np.array(trainLabels)

print('train shape: '+str(trainData.shape))
#print(str(trainData))
#print('labels shape: '+ str(trainLabelsNp.shape))
#print(str(trainLabelsNp))

hist = model.fit(normTrainData, trainLabelsCat, epochs=5000)

y_pred = model.predict_classes(normTestData)

cm = confusion_matrix(testLabels, y_pred)
print()
print('Test confusion Matrix')
print_cm(cm)
#print(cm)
cr = classification_report(testLabels, y_pred)
print(cr)
