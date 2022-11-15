import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np 
from model import model_arch

mnist = tf.keras.datasets.fashion_mnist


(trainX, trainY), (testX, testY) = mnist.load_data()

'''
Disse linjer viser billeder af de forskellige data. Et eksempel er billedet "Eksempel.png" under mappen billeder
plt.imshow(trainX[0], cmap=plt.get_cmap('gray'))
plt.show()
''' 

trainX = np.expand_dims(trainX, 1)
testX = np.expand_dims(testX, 1)

model = model_arch()

model.compile(tf.keras.optimizers.Adam(learning_rate=1e-3), loss="sparse_categorical_crossentropy", metrics=['sparse_categorical_accuracy'])

Traning_historie = model.fit(trainX.astype(np.float32), trainY.astype(np.float32), epochs=10, steps_per_epoch=100, validation_split=0.33)

model.save_weights("./model.h5", overwrite=True)
