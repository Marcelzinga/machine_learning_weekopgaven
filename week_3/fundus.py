from uitwerkingen import *
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import pathlib
import matplotlib.pyplot as plt
from random import randint
import sys


def plotMatrix(data):
    plt.matshow(data)
    plt.show()

print ("Laden van de data...")
batch_size = 1000
img_height = 75
img_width = 75

# get the current directory
current_directory = os.path.join(os.path.dirname(__file__))
# create the path to the data directory within the current directory
data_dir = os.path.join(current_directory, "Fundus-data")


#TODO uit Fundus-data de eerste 80 % train data de overgebleven 20 test data
train_ds = keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

for image_batch, labels_batch in train_ds.take(1): # todo kijk hier nog eens naar
    print("formaat van de train_ds batch" + str(image_batch.shape))
    print(labels_batch.shape)
    train_images = image_batch.numpy()
    train_labels = labels_batch.numpy()

for image_batch, labels_batch in train_ds.take(2): # todo kijk hier nog eens naar
    print("formaat train_ds.take(2)" + str(image_batch.shape))
    print(labels_batch.shape)
    val_images = image_batch.numpy()


class_names = train_ds.class_names

rnd = randint(0, 800) #train_images.shape[0])
#Plot een van de images
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    plt.imshow(images[rnd].numpy().astype("uint8"))
    plt.title(class_names[labels[rnd]])
    plt.axis("off")

    training_data = images.numpy()
    train_labels = labels.numpy()

plt.show()

for image_batch, labels_batch in train_ds:
    print("formaat van de train_ds batch" + str(image_batch.shape))
    print(labels_batch.shape)
    break


for image_batch, labels_batch in val_ds:
    print("formaat van de val_ds batch" + str(image_batch.shape))
    print(labels_batch.shape)
    break

#TODO build model en train
#TODO kies activation+optimizer+loss+metrics welke passen bij deze dataset? waarom?
# model = keras.Sequential([
#     keras.layers.Reshape((75 * 75 * 3,), input_shape=(75, 75, 3)),
#     keras.layers.experimental.preprocessing.Rescaling(1./255),  # scale the data
#     keras.layers.Dense(512, activation=tf.nn.relu),
#     keras.layers.Dense(len(class_names), activation=tf.nn.softmax)
# ])


model = tf.keras.Sequential([
    keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(75, 75, 3)),
    keras.layers.Conv2D(16, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(len(class_names))
])

# model = keras.Sequential([
#     keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(75, 75, 1)),  # scale the data
#
#     keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
#     keras.layers.MaxPooling2D(),
#     keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
#     keras.layers.MaxPooling2D(),
#     keras.layers.Flatten(),
#     keras.layers.Dense(128, activation='relu'),
#
#     #keras.layers.Dense(512, activation=tf.nn.relu),
#     keras.layers.Dense(len(class_names), activation=tf.nn.softmax)
# ])

model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


#train_images = keras.utils.to_categorical(train_images)
#train_labels = keras.utils.to_categorical(train_labels)

# model.fit(
#     train_images,
#     train_labels,
#     epochs=5,
#     #batch_size=128,
# )

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30
)


#TODO test model
pred = np.argmax(model.predict(train_ds), axis=1)
cm = confMatrix(train_labels, pred)
data = cm.numpy()
print("De confusion matrix:")

# Er wordt alleen 29 gepredict. dus is dat het enige wat true positive is #TODO verbeter het netwerk?
plotMatrix(data)

print(data)
print(data.shape)

print("Bepalen van de tp, tn, fp, fn")
metrics = confEls(data, train_labels)
print(metrics)
print("Bepalen van de scores:")
scores = confData(metrics)
print(scores)
