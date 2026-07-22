import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Dense, Input
import neural_utility_functions as nuf

X_train = np.array([[1.0], [2.0]], dtype=np.float32)           #(size in 1000 square feet)
Y_train = np.array([[300.0], [500.0]], dtype=np.float32)       #(price in 1000s of dollars)

fig, ax = plt.subplots(1,1)
ax.scatter(X_train, Y_train, marker='x', c='r', label="Data Points")
ax.legend( fontsize='xx-large')
ax.set_ylabel('Price (in 1000s of dollars)', fontsize='xx-large')
ax.set_xlabel('Size (1000 sqft)', fontsize='xx-large')
plt.show()

linear_layer = tf.keras.layers.Dense(units=1, activation = 'linear', )
print(linear_layer.get_weights()) #There are no weights as the weights are not yet instantiated.



a1 = linear_layer(X_train[0].reshape(1,1)) # X_train[0] = [1.0] has shape (1,) but TensorFlow Dense layers expect input in shape (1,1). 
# So we convert [1.0] to [[1.0]] which has shape (1,1)

print(a1) # tf.Tensor([[1.244989]], shape=(1, 1), dtype=float32)
#The result is a tensor (another name for an array) with a shape of (1,1) or one entry.

w, b= linear_layer.get_weights()
print(f"w = {w}, b={b}") #These weights are randomly initialized to small numbers and the bias defaults to being initialized to zero.

set_w = np.array([[200]]) #The weights are initialized to random values so let's set them to some known values.
set_b = np.array([100])

# set_weights takes a list of numpy arrays
linear_layer.set_weights([set_w, set_b])
print(linear_layer.get_weights())

a1 = linear_layer(X_train[0].reshape(1,1))
print(a1)
alin = np.dot(set_w,X_train[0].reshape(1,1)) + set_b  # A linear regression model with a single input feature will have a single weight and bias.
print(alin)

prediction_tf = linear_layer(X_train)  # Tensorflow prediction
prediction_np = np.dot(X_train, set_w) + set_b  # NumPy prediction

nuf.plt_linear(X_train, Y_train, prediction_tf, prediction_np)