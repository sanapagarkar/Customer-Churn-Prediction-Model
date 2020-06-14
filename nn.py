#Part 1 - Data preprocessing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset= pd.read_csv('Churn_Modelling.csv')
x=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x1 = LabelEncoder()
x[:, 1] = labelencoder_x1.fit_transform(x[:, 1])
labelencoder_x2 = LabelEncoder()
x[:, 2] = labelencoder_x2.fit_transform(x[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
x = onehotencoder.fit_transform(x).toarray()
x=x[:,1:] #removing extra dummy variable

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
x_train=sc_X.fit_transform(x_train)
x_test=sc_X.transform(x_test)

#Part 2 - Make ANN

#import keras lib and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initialising ANN
classifier = Sequential() #classifier is future NN

#Adding the input layer and first hidden layer
classifier.add(Dense(units=6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

#Adding the second hidden layer
classifier.add(Dense(units=6, kernel_initializer = 'uniform', activation = 'relu'))

#Adding the output layer
classifier.add(Dense(units=1, kernel_initializer = 'uniform', activation = 'sigmoid'))

#Compiling ANN i.e. apply SGD to ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

#Fitting ANN to training set
classifier.fit(x_train, y_train, batch_size = 10, nb_epoch = 100)

y_pred = classifier.predict(x_test) #returns probability of it being 1

y_pred = (y_pred>0.5) #returns true or false

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
