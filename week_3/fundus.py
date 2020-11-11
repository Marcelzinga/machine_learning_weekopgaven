from uitwerkingen import *
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import matplotlib.pyplot as plt
from random import randint


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

for image_batch, labels_batch in val_ds.take(1):
    print("formaat van de val_ds batch" + str(image_batch.shape))
    print(labels_batch.shape)
    val_images = image_batch.numpy()
    val_labels = labels_batch.numpy()

class_names = train_ds.class_names

rnd = randint(0, 800) #train_images.shape[0])
#Plot een van de images
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    plt.imshow(images[rnd].numpy().astype("uint8"))
    plt.title(class_names[labels[rnd]])
    plt.axis("off")

plt.show()


#TODO build model en train
#TODO kies activation+optimizer+loss+metrics welke passen bij deze dataset? waarom?
# model = keras.Sequential([
#     keras.layers.Reshape((75 * 75 * 3,), input_shape=(75, 75, 3)),
#     keras.layers.experimental.preprocessing.Rescaling(1./255),  # scale the data
#     keras.layers.Dense(512, activation=tf.nn.relu),
#     keras.layers.Dense(len(class_names), activation=tf.nn.softmax)
# ])


model = tf.keras.Sequential([
    #keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(75, 75, 3)), #dit zorgt ervoor dat alles 29 word?
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
from skimage import io
#gray_training_data = train_ds.map(lambda x, y: (io.imread(x, pilmode='L'), y))


model.fit(
    train_images,
    train_labels,
    validation_data=val_ds,
    epochs=25 #35 #45 ?
)


print("Predict met bekende data, zal hoge accuratesse hebben")
pred = np.argmax(model.predict(train_images), axis=1)
cm = confMatrix(train_labels, pred)
data = cm.numpy()
print("De confusion matrix:")

plotMatrix(data)
print(data)

print("Predict met validatie data")
pred = np.argmax(model.predict(val_images), axis=1)
cm = confMatrix(val_labels, pred)
data = cm.numpy()
print("De confusion matrix:")

# Er wordt veel 29 gepredict. #TODO verbeter het netwerk?
plotMatrix(data)
print(data)

print("Bepalen van de tp, tn, fp, fn")
metrics = confEls(data, train_labels)
print(metrics)
print("Bepalen van de scores:")
scores = confData(metrics)
print(scores)
