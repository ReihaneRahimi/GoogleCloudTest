# univariate data preparation
# 1d cnn example
# Training a model to predict a sequence of data.

from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import matplotlib.pyplot as plt
import time
import pickle

start_time = time.time()

#to load the data
pickle_in = open("data.pickle","rb")
X, y = pickle.load(pickle_in)


# choose a number of time steps based on the data preparation
n_steps = 3


# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
# fit model
History = model.fit(X, y, epochs=100, verbose=0, validation_split=0.1)
# demonstrate prediction

model.save ('TimeSeriesTest1.model')


x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print("yhat = ", yhat)

# print ("Duration =", time.time() - start_time)

# print(History.history.keys())


# summarize history for loss
plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('plot.pdf', dpi=300, bbox_inches='tight')

plt.show()







