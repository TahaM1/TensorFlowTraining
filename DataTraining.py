import tensorflow as tf
import numpy as np

tf.enable_eager_execution()

print(tf.add(1, 2))
print(tf.add([3, 4], [1, 2]))
print(tf.reduce_sum([1, 2, 3]))

array = np.ones([3, 3])
tensor = tf.multiply(array, 42)
print(tensor)

print(np.add(tensor, 1))
print(tensor.numpy())
