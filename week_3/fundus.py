from uitwerkingen import *
from tensorflow import keras
import numpy as np
import os
import pathlib

print ("Laden van de data...")
batch_size = 1000
img_height = 32
img_width = 32

# get the current directory
current_directory = os.path.join(os.path.dirname(__file__))
print(current_directory)
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

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    #plt.title(class_names[labels[i]])
    plt.axis("off")

plt.show()


#TODO build model

#TODO test model