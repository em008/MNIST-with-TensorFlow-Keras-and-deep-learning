"""
Parameters
The batch size, number of training epochs and location of the data files is defined here. 
Data files are hosted in a Google Cloud Storage (GCS) bucket which is why their address starts with gs://.
"""
BATCH_SIZE = 128
EPOCHS = 10

training_images_file   = 'gs://mnist-public/train-images-idx3-ubyte'
training_labels_file   = 'gs://mnist-public/train-labels-idx1-ubyte'
validation_images_file = 'gs://mnist-public/t10k-images-idx3-ubyte'
validation_labels_file = 'gs://mnist-public/t10k-labels-idx1-ubyte'

"""
Imports
All the necessary Python libraries are imported here, including TensorFlow.
"""
import os, re, math, json, shutil, pprint
import numpy as np
import tensorflow as tf
print("Tensorflow version " + tf.__version__)

"""
Parse files and prepare training and validation datasets
"""
AUTO = tf.data.experimental.AUTOTUNE

def read_label(tf_bytestring):
    label = tf.io.decode_raw(tf_bytestring, tf.uint8)
    label = tf.reshape(label, [])
    label = tf.one_hot(label, 10)
    return label
  
def read_image(tf_bytestring):
    image = tf.io.decode_raw(tf_bytestring, tf.uint8)
    image = tf.cast(image, tf.float32)/256.0
    image = tf.reshape(image, [28*28])
    return image
  
"""
Apply this function to the dataset using .map and obtain a dataset of images.
The same kind of reading and decoding for is done using .zip for images and labels.
"""
def load_dataset(image_file, label_file):
    imagedataset = tf.data.FixedLengthRecordDataset(image_file, 28*28, header_bytes=16)
    imagedataset = imagedataset.map(read_image, num_parallel_calls=16)
    labelsdataset = tf.data.FixedLengthRecordDataset(label_file, 1, header_bytes=8)
    labelsdataset = labelsdataset.map(read_label, num_parallel_calls=16)
    dataset = tf.data.Dataset.zip((imagedataset, labelsdataset))
    return dataset 
  
"""
Training dataset
The tf.data.Dataset API has all the necessary utility functions for preparing datasets.
.cache caches the dataset in RAM
.shuffle shuffles it with a buffer of 5000 elements
.repeat loops the dataset
.batch pulls multiple images and labels together into a mini-batch
.prefetch can use the CPU to prepare the next batch while the current batch is being trained on the GPU
The validation dataset is prepared in a similar way
"""
def get_training_dataset(image_file, label_file, batch_size):
    dataset = load_dataset(image_file, label_file)
    dataset = dataset.cache()  # this small dataset can be entirely cached in RAM, for TPU this is important to get good performance from such a small dataset
    dataset = dataset.shuffle(5000, reshuffle_each_iteration=True)
    dataset = dataset.repeat() # Mandatory for Keras for now
    dataset = dataset.batch(batch_size, drop_remainder=True) # drop_remainder is important on TPU, batch size must be fixed
    dataset = dataset.prefetch(AUTO)  # fetch next batches while training on the current one (-1: autotune prefetch buffer size)
    return dataset
  
def get_validation_dataset(image_file, label_file):
    dataset = load_dataset(image_file, label_file)
    dataset = dataset.cache() # this small dataset can be entirely cached in RAM, for TPU this is important to get good performance from such a small dataset
    dataset = dataset.batch(10000, drop_remainder=True) # 10000 items in eval dataset, all in one batch
    dataset = dataset.repeat() # Mandatory for Keras for now
    return dataset

# instantiate the datasets
training_dataset = get_training_dataset(training_images_file, training_labels_file, BATCH_SIZE)
validation_dataset = get_validation_dataset(validation_images_file, validation_labels_file)

# For TPU, we will need a function that returns the dataset
training_input_fn = lambda: get_training_dataset(training_images_file, training_labels_file, BATCH_SIZE)
validation_input_fn = lambda: get_validation_dataset(validation_images_file, validation_labels_file)

"""
Keras Model
The models will be straight sequences of layers so tf.keras.Sequential is used to create them. 
There are 10 neurons because we are classifying handwritten digits into 10 classes.
It uses "softmax" activation because it is the last layer in a classifier.
A Keras model also needs to know the shape of its inputs, tf.keras.layers.Input can be used to define it.
Configuring the model is done in Keras using the model.compile function. 
Here we use the basic optimizer 'sgd' (Stochastic Gradient Descent). 
A classification model requires a cross-entropy loss function, called 'categorical_crossentropy' in Keras. 
The model computes the 'accuracy' metric, which is the percentage of correctly classified images.
"""
model = tf.keras.Sequential(
  [
      tf.keras.layers.Input(shape=(28*28,)),
      tf.keras.layers.Dense(10, activation='softmax')
  ]
)

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# print model layers
model.summary()

"""
"Train and validate the model"
The training happens by calling model.fit and passing in both the training and validation datasets.
"""
steps_per_epoch = 60000//BATCH_SIZE  # 60,000 items in this dataset
print("Steps per epoch: ", steps_per_epoch)

history = model.fit(training_dataset, steps_per_epoch=steps_per_epoch, epochs=EPOCHS, validation_data=validation_dataset, validation_steps=1)

"""
The shape of the output tensor is [128, 10] because 128 images are processed and computing the argmax across the 10 probabilities returned for each image, thus axis=1.
This model recognises approximately 90% of the digits.
"""
