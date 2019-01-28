# pip install theano
# pip install tensorflow
# pip install keras
# pip install rope_py3k (optional) used by spyder to autocomplete

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data

# transform country into numbers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

# transform gender into numbers
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# indexes
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()

# delete dummy data thing ???
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initializing the ANN
classifier = Sequential()  # create the model, but with no layers ()

# activation = 'relu' = rectifier function
# activation = 'sigmoid' = sigmoid function
# activation = 'softmax' = sigmoid function for category > 1 (units = 1)
# kernel_initializer = 'uniform' = initialize weights uniform and close to 0

# Input layer and first layer (input layer) and the first hidden layer with dropout
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
classifier.add(Dropout(p=0.1))  # Dropout added to the first layer, this diables 10% of the nodes at each iteration

# Adding the second hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(p=0.1))  # Dropout added to the second layer, this disables 10% of the nodes at each iteration

# Adding the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compile the ANN
# optimizer = 'adam' = stochastic gradient descent algorithm, optimizer function
# loss = depends on the output, if we output binary than we use binary_crossentropy else another function
# metrics
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# epochs = number of times for training the ANN

# Fitting classifier to the Training set

# X_train = is the set with the train inputs
# y_train = is the set with the train results

classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)  # this returns probabilities

y_pred = (y_pred > 0.5)  # convert from probabilities to 1 or 0

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

# Predict if the customer with the following informations will leave the bank:

"""
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000

"""

new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)

# Evaluating the ANN

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense


def build_classifier():
    # Initializing the ANN
    classifier = Sequential()  # create the model, but with no layers ()

    # activation = 'relu' = rectifier function
    # activation = 'sigmoid' = sigmoid function
    # activation = 'softmax' = sigmoid function for category > 1 (units = 1)
    # kernel_initializer = 'uniform' = initialize weights uniform and close to 0

    # Input layer and first layer (input layer) and the first hidden layer
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))

    # Adding the second hidden layer
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

    # Adding the output layer
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

    # Compile the ANN
    # optimizer = 'adam' = stochastic gradient descent algorithm, optimizer function
    # loss = depends on the output, if we output binary than we use binary_crossentropy else another function
    # metrics
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=-1)
mean = accuracies.mean()
variances = accuracies.std()

# Improving the ANN
# Dropout Regularization to reduce overfitting if needed
# from keras.layers import Dropout
# classifier.add(Dropout(p = 0.1)) # Dropout added to the first layer, this diables 10% of the nodes at each iteration


# Tuning the ANN - tunning the batch size and epoch
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense


def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


classifier = KerasClassifier(build_fn=build_classifier)

parameters = {
    'batch_size': [25, 32],  # test batch_size with those values
    'nb_epoch': [100, 500],  # epoch tests
    'optimizer': ['adam', 'rmsprop']  # test different stochastic gradient descent algorithms, optimizer function
}

grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
