from uitwerkingen import *
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import matplotlib.pyplot as plt
from random import randint


def plotMatrix(cm_traindata, cm_valdata2):
    fig, axes = plt.subplots(1, 2)
    axes[0].matshow(cm_traindata)
    axes[0].set_title("Training data")
    axes[1].matshow(cm_valdata2)
    axes[1].set_title("Validation data")
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
# model = keras.Sequential([
#     keras.layers.Reshape((75 * 75 * 3,), input_shape=(75, 75, 3)),
#     keras.layers.experimental.preprocessing.Rescaling(1./255),  # scale the data
#     keras.layers.Dense(512, activation=tf.nn.relu),
#     keras.layers.Dense(len(class_names), activation=tf.nn.softmax)
# ])


model = tf.keras.Sequential([
    #keras.layers.experimental.preprocessing.Rescaling(1./255), #dit zorgt ervoor dat alles 29 word?
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


train_images = tf.image.rgb_to_grayscale(train_images)
val_images = tf.image.rgb_to_grayscale(val_images)
# train_images = train_images.numpy().reshape(800, 75, 75)
# val_images = val_images.numpy().reshape(200, 75, 75)


epochs = 20
history = model.fit(
    train_images,
    train_labels,
    validation_data=(val_images, val_labels),
    epochs=epochs,
    batch_size=128
)
model.summary()
print("evaluate model")
test_loss, test_acc = model.evaluate(val_images, val_labels)
print('test_acc:', test_acc)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

print("Predict met bekende data, zal hoge accuratesse hebben")
pred = np.argmax(model.predict(train_images), axis=1)
cm = confMatrix(train_labels, pred)
cm_traindata = cm.numpy()


print("Predict met validatie data")
pred = np.argmax(model.predict(val_images), axis=1)
cm = confMatrix(val_labels, pred)
cm_valdata = cm.numpy()

# Er wordt veel 29 gepredict. #TODO verbeter het netwerk?
plotMatrix(cm_traindata, cm_valdata)

print("Training data:")
print("Bepalen van de tp, tn, fp, fn")
metrics = confEls(cm_traindata, train_labels)
print(metrics)
print("Bepalen van de scores:")
scores = confData(metrics)
print(scores)

print("Validation data:")
print("Bepalen van de tp, tn, fp, fn")
metrics = confEls(cm_valdata, train_labels)
print(metrics)
print("Bepalen van de scores:")
scores = confData(metrics)
print(scores)