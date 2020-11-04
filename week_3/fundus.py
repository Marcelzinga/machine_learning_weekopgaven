from uitwerkingen import *
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import pathlib
import matplotlib.pyplot as plt
from random import randint

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


# train_np = np.stack(list(train_ds))
# val_np = np.stack(list(val_ds))
#
# print(train_np.shape) #verwacht 800, 75, 75
# print(val_np.shape)


rnd = randint(0, 100) #train_images.shape[0])
#Plot een van de images
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):

    plt.imshow(images[rnd].numpy().astype("uint8"))
    plt.title(class_names[labels[rnd]])
    plt.axis("off")

plt.show()

for image_batch, labels_batch in train_ds:
    print("formaat van de train_ds batch" + str(image_batch.shape))
    print(labels_batch.shape)
    break

for image_batch, labels_batch in val_ds:
    print("formaat van de val_ds batch" + str(image_batch.shape))
    print(labels_batch.shape)
    break

# ScaleData
normalization_layer = keras.layers.experimental.preprocessing.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

#TODO build model en train
#TODO kies activation+optimizer+loss+metrics welke passen bij deze dataset? waarom?
model = keras.Sequential([
    keras.layers.Reshape((75 * 75 * 3,), input_shape=(75, 75, 3)),
    keras.layers.experimental.preprocessing.Rescaling(1./255),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(len(class_names), activation=tf.nn.softmax)

])

model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3
)


#TODO test model
pred = np.argmax(model.predict(train_ds), axis=1)

#cm = confMatrix(class_names, pred)
#data = cm.numpy()
# print("De confusion matrix:")
# if (len(sys.argv) > 1 and sys.argv[1] == 'skip'):
#     print("Tekenen slaan we over")
# else:
#     plotMatrix(data)
#
# print(data)
# print(data.shape)
#
# print("Bepalen van de tp, tn, fp, fn")
# metrics = confEls(data, labels)
# print(metrics)
# print("Bepalen van de scores:")
# scores = confData(metrics)
# print(scores)
