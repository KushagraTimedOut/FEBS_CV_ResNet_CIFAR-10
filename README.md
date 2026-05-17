# Implementation of the ResNet paper (He et al., CVPR 2016)

## Files
- model.py - ResNet architecture
- dataset.py - CIFAR-10 loading, normalization, and data augmentation
- train.py - training, evaluation, and plots
- requirements.txt

## How to run
Install dependencies:

    pip install -r requirements.txt
Train:

    python train.py

CIFAR-10 downloads automatically. After training, loss and accuracy curves pop up in a matplotlib window.

## Hyperparameters
- Optimizer: SGD
- Momentum: 0.9
- Weight decay: 1e-4
- Learning rate: 0.1
- Batch size: 128
- Epochs: 200

## Architecture
ResNet-20 for 32x32 images. Three stages with 16, 32, and 64 filters, each with 3 residual blocks. Ends with global average pooling and a dense softmax layer. Total number of parameters is around 270000

## Data augmentation
Images are normalized per channel. Training uses random horizontal flips and random shifts of up to 4 pixels to simulate the padding and cropping from the paper. No augmentation on validation as it would cause evaluation metrics to be influenced by the artificially created data.

## Expected accuracy
Around 89-90% on the test set after 200 epochs.
