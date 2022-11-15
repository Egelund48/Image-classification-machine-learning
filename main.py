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

trainX = np.expand_dims(trainX, -1)
testX = np.expand_dims(testX, -1)

model = model_arch()

model.compile(tf.keras.optimizers.Adam(learning_rate=1e-3), loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])

#model.summary()

#Traning_historie=model.fit(trainX.astype(np.float32), trainY.astype(np.float32), epochs=20, steps_per_epoch=100, validation_steps=0.33)

model.save_weights("./model.h5", overwrite=True)

'''
plt.plot(Traning_historie.history['sparse_categorical_accuracy'])
plt.plot(Traning_historie.history['val_sparse_categorical_accuracy'])
plt.title('Modellens nøjagtighed')
plt.ylabel('Nøjagtighed')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
'''
labels = ["T-shirt", "Bukser", "Pullover", "Kjole", "Frakke", "Sandal", "Skjorte", "Sneaker", "Taske", "Ankel sko"]

predictions = model.predict(testX[:1])
label = labels[np.argmax(predictions)]
 
print(label)
plt.imshow(testX[:1][0])
plt.show()