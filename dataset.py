import numpy as np
import keras
from keras import layers
from keras.src.legacy.preprocessing.image import ImageDataGenerator

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def load_and_preprocess_cifar10():
    
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    x_train = x_train.astype('float32') / 255.0
    x_test  = x_test.astype('float32')  / 255.0

    mean = x_train.mean(axis=(0, 1, 2))
    std  = x_train.std(axis=(0, 1, 2))

    x_train = (x_train - mean) / std
    x_test  = (x_test  - mean) / std

    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test  = keras.utils.to_categorical(y_test,  num_classes)

    val_size = 5000
    x_val   = x_train[-val_size:]
    y_val   = y_train[-val_size:]
    x_train = x_train[:-val_size]
    y_train = y_train[:-val_size]

    return x_train, y_train, x_val, y_val, x_test, y_test, mean, std


def get_data_generators(x_train, y_train, x_val, y_val, batch_size=128):
    train_datagen = ImageDataGenerator(
        width_shift_range=4/32,       # shift up to 4 pixels horizontally
        height_shift_range=4/32,      # shift up to 4 pixels vertically
        horizontal_flip=True,         # random left-right flip
        fill_mode='reflect'           # pad edges by reflection (simulates crop)
    )

    val_datagen = ImageDataGenerator()

    train_gen = train_datagen.flow(
        x_train, y_train,
        batch_size=batch_size,
        shuffle=True
    )
    val_gen = val_datagen.flow(
        x_val, y_val,
        batch_size=batch_size,
        shuffle=False
    )

    return train_gen, val_gen

