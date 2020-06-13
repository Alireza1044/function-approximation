import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras

train_data_count = 10000
test_data_count = 200

x_train = np.random.randint(low=-100000, high=300000, size=train_data_count) / 100
y_train = np.power(x_train, 2)

x_test = np.random.randint(low=-1000, high=3000, size=test_data_count) / 100
y_test = np.power(x_test, 2)

test = np.linspace(-3, 3, 60)
test = np.reshape(test, 60, 1)

x_train = np.reshape(x_train, (train_data_count, 1))
y_train = np.reshape(y_train, (train_data_count, 1))
x_test = np.reshape(x_test, (test_data_count, 1))
y_test = np.reshape(y_test, (test_data_count, 1))

keras.utils.normalize(x_train)
keras.utils.normalize(y_train)

model = Sequential()
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))
sgd = keras.optimizers.sgd(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='adam', loss='mean_squared_logarithmic_error', metrics=['mean_squared_logarithmic_error'])
model.fit(x_train, y_train, batch_size=10, validation_data=(x_test, y_test), epochs=20)
model.evaluate(x_test, y_test)
model.test_on_batch(x_test, y_test)
model.predict(test)

plt.plot(test)
plt.plot(np.power(test,2))
plt.show()

