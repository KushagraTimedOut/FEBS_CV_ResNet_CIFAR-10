import tensorflow as tf
import keras
from keras import layers

def residual_block(x, filters, stride=1, downsample=False):
    shortcut = x

    x = layers.Conv2D(filters, kernel_size=3, strides=stride,
                      padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, kernel_size=3, strides=1,
                      padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if downsample:
        shortcut = layers.Conv2D(filters, kernel_size=1, strides=stride,
                                 padding='same', use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)

    return x

# ResNet-20 for CIFAR-10 (32×32 input)

def build_resnet_cifar10(num_classes=10):
    """
    Architecture of the model:
      - Input: 32×32×3
      - Initial 3×3 conv, 16 filters
      - 3 stages of residual blocks with {16, 32, 64} filters
      - Global Average Pooling → Dense(10) → Softmax

    Total layers: 6n+2 where n=3 => 20 layers
    """
    inputs = keras.Input(shape=(32, 32, 3), name='input_image')

    x = layers.Conv2D(16, kernel_size=3, strides=1,
                      padding='same', use_bias=False, name='conv_initial')(inputs)
    x = layers.BatchNormalization(name='bn_initial')(x)
    x = layers.ReLU(name='relu_initial')(x)

    for i in range(3):
        x = residual_block(x, filters=16, stride=1, downsample=False)

    x = residual_block(x, filters=32, stride=2, downsample=True)
    for i in range(2):
        x = residual_block(x, filters=32, stride=1, downsample=False)

    x = residual_block(x, filters=64, stride=2, downsample=True)
    for i in range(2):
        x = residual_block(x, filters=64, stride=1, downsample=False)

    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = keras.Model(inputs, outputs, name='ResNet20_CIFAR10')
    return model    
