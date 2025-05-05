#MNIST Handwritten Digit data from Kaggle

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Ignore the warnings
import warnings
warnings.filterwarnings("ignore")

#loading MNIST dataset
from tensorflow.keras.datasets import mnist
(X_train,y_train) , (X_test,y_test)=mnist.load_data()

#visualizing the image in train data
plt.imshow(X_train[0])

#visualizing the first 20 images in the dataset
for i in range(20):

    #subplot
    plt.subplot(5, 5, i+1)

    # plotting pixel data
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))

# show the figure
plt.show()

print(X_train.shape)
print(X_test.shape)

# the image is in pixels which ranges from 0 to 255
X_train[0]
X_train_flat=X_train.reshape(len(X_train),28*28)
X_test_flat=X_test.reshape(len(X_test),28*28)

#checking the shape after flattening
print(X_train_flat.shape)
print(X_test_flat.shape)

#checking the representation of image after flattening
X_train_flat[0]

#normalizing the pixel values
X_train_flat=X_train_flat/255
X_test_flat=X_test_flat/255
#checking the pixel values after normalization
X_train_flat[0]

"""Building a deep learning model is built in the following steps:

1.   Defining the model
2.   Compiling the model
3.   Fitting the model
4.   Evaluating the model
5.   Making Predictions


"""

#Building a simple ANN model without hidden layer
#importing necessary libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#Step 1 : Defining the model
model=Sequential()
model.add(Dense(10,input_shape=(784,),activation='softmax'))

#Step 2: Compiling the model
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#Step 3: Fitting the model
model.fit(X_train_flat,y_train,epochs=10)

#Step 4: Evaluating the model
model.evaluate(X_test_flat,y_test)

#Step 5 :Making predictions
y_predict = model.predict(X_test_flat)
y_predict[3] #printing the 3rd index

# Here we get the index of the maximum value in the above-encoded vector.
np.argmax(y_predict[3])

#checking if the predicting is correct
plt.imshow(X_test[3])


#Plotting the confusion matrix to check the Actual VS Predicted Labels
y_predict_labels=np.argmax(y_predict,axis=1)
#Confusion matrix
from sklearn.metrics import confusion_matrix
matrix=confusion_matrix(y_test,y_predict_labels)
#visualizaing confusion matrix with heatmap
plt.figure(figsize=(10,7))
sns.heatmap(matrix,annot=True,fmt='d')