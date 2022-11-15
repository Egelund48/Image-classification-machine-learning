import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np 

def model_arch(): 
    models = tf.keras.models.Sequential()

    models.add(tf.keras.layers.Conv2D(64, (5,5), padding="same", activation="relu", input_shape=(28,28,1)))

    models.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    models.add(tf.keras.layers.Conv2D(256,(5,5), padding="same", activation="relu"))

    models.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

    models.add(tf.keras.layers.Flatten())
    models.add(tf.keras.layers.Dense(256,activation="relu"))

    models.add(tf.keras.layers.Dense(10, activation="softmax"))

    return models 