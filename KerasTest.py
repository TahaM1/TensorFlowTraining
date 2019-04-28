import tensorflow as tf
from tensorflow._api.v1.keras import layers
import numpy as np

print(tf.VERSION)
print(tf.keras.__version__)


model = tf.keras.Sequential()

# adding layers
model.add(layers.Dense(64, activation='relu'))  # layer with 64 units
model.add(layers.Dense(64, activation='relu'))  # layer with 64 units
model.add(layers.Dense(10, activation='softmax'))  # 10 output units

# sigmoid layer
layers.Dense(64, activation='sigmoid')

# an Initializer creates weights for the kernel and bias while
# the regularizer applys the layer weights

# linear layer with L1 regularization of 0.01 applied to kernel matrix
layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.01))

# linear layer with l2 regularization of 0.01 applied to bias vector
layers.Dense(64, bias_regularizer=tf.keras.regularizers.l2(0.01))

# linear layer with a kernal initialized to a random orthogonal matrix
layers.Dense(64, kernel_initializer='orthogonal')

# linear layer with a bias vector initialized to 2.0s:
layers.Dense(64, bias_initializer=tf.keras.initializers.constant(2.0))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

model = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(32,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# compile has 3 important parameters optimizer, loss and metrics
model.compile(optimizer=tf.train.AdamOptimizer(0.001),  # optimizer specifies training procedure
              loss='categorical_crossentropy',  # loss is the function to minimize during optimization
              metrics=['accuracy'])  # used to monitor training

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

val_data = np.random.random((100, 32))
val_labels = np.random.random((100, 10))


# 3 main parameters epochs, batch_size, validation_data

# epochs is the amount of iteration the data will train
# the model slices up the data into small batches of batch_size
# validation_data allows the model to display the loss and metrics in inference
#   mode for the passed data, at the end of each epoch

# model.fit(data, labels, epochs=10, batch_size=32,
#           validation_data=(val_data, val_labels))

dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)
dataset = dataset.repeat()

model.fit(dataset, epochs=10, steps_per_epoch=30)


