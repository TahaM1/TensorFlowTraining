import tensorflow as tf
import numpy as np

tf.enable_eager_execution()

# print(tf.add(1, 2))
# print(tf.add([3, 4], [1, 2]))
# print(tf.reduce_sum([1, 2, 3]))
#
# array = np.ones([3, 3])
# tensor = tf.multiply(array, 42)
# print(tensor)
#
# print(np.add(tensor, 1))
# print(tensor.numpy())

ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])

ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])

# Create a CSV file
import tempfile
_, filename = tempfile.mkstemp()

with open(filename, 'w') as f:
    f.write("""Line 1
Line 2
Line 3
    """)
ds_file = tf.data.TextLineDataset(filename)

ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)

ds_file = ds_file.batch(2)

print("Elements of ds_tensors: ")
for x in ds_tensors:
    print(x)

print("\nElements of ds_file")
for x in ds_file:
    print(x)
