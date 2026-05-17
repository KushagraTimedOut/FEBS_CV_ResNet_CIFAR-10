import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras import optimizers

from model   import build_resnet_cifar10
from dataset import load_and_preprocess_cifar10, get_data_generators

BATCH_SIZE   = 128
EPOCHS       = 200          # paper uses ~64k iterations; ~200 epochs here
INITIAL_LR   = 0.1
WEIGHT_DECAY = 1e-4
NUM_CLASSES  = 10


def plot_curves(history):
    epochs = range(1, len(history['loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('ResNet CIFAR-10 Training Curves', fontsize=14)

    # Loss 
    ax1.plot(epochs, history['loss'],     'b-',  linewidth=1.5, label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r--', linewidth=1.5, label='Val Loss')
    for drop in [100, 150]:
        ax1.axvline(x=drop, color='gray', linestyle=':', alpha=0.7,
                    label=f'LR drop (epoch {drop})')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history['accuracy'],     'b-',  linewidth=1.5, label='Train Accuracy')
    ax2.plot(epochs, history['val_accuracy'], 'r--', linewidth=1.5, label='Val Accuracy')
    for drop in [100, 150]:
        ax2.axvline(x=drop, color='gray', linestyle=':', alpha=0.7,
                    label=f'LR drop (epoch {drop})')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

tf.random.set_seed(42)
np.random.seed(42)

x_train, y_train, x_val, y_val, x_test, y_test, _, _ = \
    load_and_preprocess_cifar10()

train_gen, val_gen = get_data_generators(
    x_train, y_train, x_val, y_val, batch_size=BATCH_SIZE
)

steps_per_epoch  = len(x_train) // BATCH_SIZE
validation_steps = len(x_val)   // BATCH_SIZE

model = build_resnet_cifar10(num_classes=NUM_CLASSES)

optimizer = optimizers.SGD(
    learning_rate=INITIAL_LR,
    momentum=0.9,
    weight_decay=WEIGHT_DECAY
)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=val_gen,
    validation_steps=validation_steps,
    verbose=1
)

test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0)
print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_acc * 100:.2f}%")

plot_curves(history.history)