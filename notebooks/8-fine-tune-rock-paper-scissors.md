---
title: 'Fine tune a network on rock paper scissors data'
author: 'Fraida Fund'
jupyter:
  accelerator: GPU
  colab:
    name: '8-fine-tune-rock-paper-scissors.ipynb'
    toc_visible: true
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: 'text/x-python'
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.7.6
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .markdown }
# Demo: Transfer learning

*Fraida Fund*
:::

::: {.cell .markdown }
In practice, for most machine learning problems, you wouldn't design or
train a convolutional neural network from scratch - you would use an
existing model that suits your needs (does well on ImageNet, size is
right) and fine-tune it on your own data.
:::

::: {.cell .markdown }
Note: for faster training, use Runtime \> Change Runtime Type to run
this notebook on a GPU.
:::

::: {.cell .markdown }
## Import dependencies
:::

::: {.cell .code }
```python
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import platform
import datetime
import os
import math
import random

print('Python version:', platform.python_version())
print('Tensorflow version:', tf.__version__)
print('Keras version:', tf.keras.__version__)
```
:::

::: {.cell .markdown }
## Import data
:::

::: {.cell .markdown }
The "rock paper scissors" dataset is available directly from the
Tensorflow package. In the cells that follow, we'l get the data, plot
a few examples, and also do some preprocessing.
:::

::: {.cell .code }
```python
import tensorflow_datasets as tfds
```
:::

::: {.cell .code }
```python
(ds_train, ds_test), ds_info = tfds.load(
    'rock_paper_scissors',
    split=['train', 'test'],
    shuffle_files=True,
    with_info=True
)
```
:::

::: {.cell .code }
```python
fig = tfds.show_examples(ds_info, ds_train)
```
:::

::: {.cell .code }
```python
classes = np.array(['rock', 'paper', 'scissors'])
```
:::

::: {.cell .markdown }
## Pre-process dataset
:::

::: {.cell .code }
```python
INPUT_IMG_SIZE = 224
INPUT_IMG_SHAPE = (224, 224, 3)
```
:::

::: {.cell .code }
```python
def preprocess_image(sample):
    sample['image'] = tf.cast(sample['image'], tf.float32)
    sample['image'] = sample['image'] / 255.
    sample['image'] = tf.image.resize(sample['image'], [INPUT_IMG_SIZE, INPUT_IMG_SIZE])
    return sample
```
:::

::: {.cell .code }
```python
ds_train = ds_train.map(preprocess_image)
ds_test  = ds_test.map(preprocess_image)
```
:::

::: {.cell .code }
```python
fig = tfds.show_examples(ds_train, ds_info, )
```
:::

::: {.cell .markdown }
We'l convert to `numpy` format again:
:::

::: {.cell .code }
```python
train_numpy = np.vstack(tfds.as_numpy(ds_train))
test_numpy = np.vstack(tfds.as_numpy(ds_test))

X_train = np.array(list(map(lambda x: x[0]['image'], train_numpy)))
y_train = np.array(list(map(lambda x: x[0]['label'], train_numpy)))

X_test = np.array(list(map(lambda x: x[0]['image'], test_numpy)))
y_test = np.array(list(map(lambda x: x[0]['label'], test_numpy)))
```
:::

::: {.cell .markdown }
## Upload custom test sample

This code expects a PNG image.
:::

::: {.cell .code }
```python
from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))
```
:::

::: {.cell .code }
```python
from PIL import Image
 
# Edit the filename here as needed
filename = 'scissors.png'
 
# pre-process image
image = Image.open(filename).convert('RGB')
image_resized = image.resize((INPUT_IMG_SIZE, INPUT_IMG_SIZE), Image.BICUBIC)
test_sample = np.array(image_resized)/255.0
test_sample = test_sample.reshape(1, INPUT_IMG_SIZE, INPUT_IMG_SIZE, 3)
```
:::

::: {.cell .code }
```python
import seaborn as sns

plt.figure(figsize=(4,4));
plt.imshow(test_sample.reshape(INPUT_IMG_SIZE, INPUT_IMG_SIZE, 3));
```
:::

::: {.cell .markdown }
## Classify with MobileNetV2
:::

::: {.cell .markdown }
[Keras Applications](https://keras.io/api/applications/) are pre-trained
models with saved weights, that you can download and use without any
additional training.
:::

::: {.cell .markdown }
Here\'s a table of the models available as Keras Applications.

In this table, the top-1 and top-5 accuracy refer to the model\'s
performance on the ImageNet validation dataset, and depth is the depth
of the network including activation layers, batch normalization layers,
etc.
:::

::: {.cell .markdown }
```{=html}
<table>
<thead>
<tr>
<th>Model</th>
<th align="right">Size</th>
<th align="right">Top-1 Accuracy</th>
<th align="right">Top-5 Accuracy</th>
<th align="right">Parameters</th>
<th align="right">Depth</th>
</tr>
</thead>
<tbody>
<tr>
<td>Xception</td>
<td align="right">88 MB</td>
<td align="right">0.790</td>
<td align="right">0.945</td>
<td align="right">22,910,480</td>
<td align="right">126</td>
</tr>
<tr>
<td>VGG16</td>
<td align="right">528 MB</td>
<td align="right">0.713</td>
<td align="right">0.901</td>
<td align="right">138,357,544</td>
<td align="right">23</td>
</tr>
<tr>
<td>VGG19</td>
<td align="right">549 MB</td>
<td align="right">0.713</td>
<td align="right">0.900</td>
<td align="right">143,667,240</td>
<td align="right">26</td>
</tr>
<tr>
<td>ResNet50</td>
<td align="right">98 MB</td>
<td align="right">0.749</td>
<td align="right">0.921</td>
<td align="right">25,636,712</td>
<td align="right">-</td>
</tr>
<tr>
<td>ResNet101</td>
<td align="right">171 MB</td>
<td align="right">0.764</td>
<td align="right">0.928</td>
<td align="right">44,707,176</td>
<td align="right">-</td>
</tr>
<tr>
<td>ResNet152</td>
<td align="right">232 MB</td>
<td align="right">0.766</td>
<td align="right">0.931</td>
<td align="right">60,419,944</td>
<td align="right">-</td>
</tr>
<tr>
<td>ResNet50V2</td>
<td align="right">98 MB</td>
<td align="right">0.760</td>
<td align="right">0.930</td>
<td align="right">25,613,800</td>
<td align="right">-</td>
</tr>
<tr>
<td>ResNet101V2</td>
<td align="right">171 MB</td>
<td align="right">0.772</td>
<td align="right">0.938</td>
<td align="right">44,675,560</td>
<td align="right">-</td>
</tr>
<tr>
<td>ResNet152V2</td>
<td align="right">232 MB</td>
<td align="right">0.780</td>
<td align="right">0.942</td>
<td align="right">60,380,648</td>
<td align="right">-</td>
</tr>
<tr>
<td>InceptionV3</td>
<td align="right">92 MB</td>
<td align="right">0.779</td>
<td align="right">0.937</td>
<td align="right">23,851,784</td>
<td align="right">159</td>
</tr>
<tr>
<td>InceptionResNetV2</td>
<td align="right">215 MB</td>
<td align="right">0.803</td>
<td align="right">0.953</td>
<td align="right">55,873,736</td>
<td align="right">572</td>
</tr>
<tr>
<td>MobileNet</td>
<td align="right">16 MB</td>
<td align="right">0.704</td>
<td align="right">0.895</td>
<td align="right">4,253,864</td>
<td align="right">88</td>
</tr>
<tr>
<td>MobileNetV2</td>
<td align="right">14 MB</td>
<td align="right">0.713</td>
<td align="right">0.901</td>
<td align="right">3,538,984</td>
<td align="right">88</td>
</tr>
<tr>
<td>DenseNet121</td>
<td align="right">33 MB</td>
<td align="right">0.750</td>
<td align="right">0.923</td>
<td align="right">8,062,504</td>
<td align="right">121</td>
</tr>
<tr>
<td>DenseNet169</td>
<td align="right">57 MB</td>
<td align="right">0.762</td>
<td align="right">0.932</td>
<td align="right">14,307,880</td>
<td align="right">169</td>
</tr>
<tr>
<td>DenseNet201</td>
<td align="right">80 MB</td>
<td align="right">0.773</td>
<td align="right">0.936</td>
<td align="right">20,242,984</td>
<td align="right">201</td>
</tr>
<tr>
<td>NASNetMobile</td>
<td align="right">23 MB</td>
<td align="right">0.744</td>
<td align="right">0.919</td>
<td align="right">5,326,716</td>
<td align="right">-</td>
</tr>
<tr>
<td>NASNetLarge</td>
<td align="right">343 MB</td>
<td align="right">0.825</td>
<td align="right">0.960</td>
<td align="right">88,949,818</td>
<td align="right">-</td>
</tr>
<tr>
<td>EfficientNetB0</td>
<td align="right">29 MB</td>
<td align="right">-</td>
<td align="right">-</td>
<td align="right">5,330,571</td>
<td align="right">-</td>
</tr>
<tr>
<td>EfficientNetB1</td>
<td align="right">31 MB</td>
<td align="right">-</td>
<td align="right">-</td>
<td align="right">7,856,239</td>
<td align="right">-</td>
</tr>
<tr>
<td>EfficientNetB2></td>
<td align="right">36 MB</td>
<td align="right">-</td>
<td align="right">-</td>
<td align="right">9,177,569</td>
<td align="right">-</td>
</tr>
<tr>
<td>EfficientNetB3</td>
<td align="right">48 MB</td>
<td align="right">-</td>
<td align="right">-</td>
<td align="right">12,320,535</td>
<td align="right">-</td>
</tr>
<tr>
<td>EfficientNetB4</td>
<td align="right">75 MB</td>
<td align="right">-</td>
<td align="right">-</td>
<td align="right">19,466,823</td>
<td align="right">-</td>
</tr>
<tr>
<td>EfficientNetB5</td>
<td align="right">118 MB</td>
<td align="right">-</td>
<td align="right">-</td>
<td align="right">30,562,527</td>
<td align="right">-</td>
</tr>
<tr>
<td>EfficientNetB6</td>
<td align="right">166 MB</td>
<td align="right">-</td>
<td align="right">-</td>
<td align="right">43,265,143</td>
<td align="right">-</td>
</tr>
<tr>
<td>EfficientNetB7</td>
<td align="right">256 MB</td>
<td align="right">-</td>
<td align="right">-</td>
<td align="right">66,658,687</td>
<td align="right">-</td>
</tr>
</tbody>
</table>
```
:::

::: {.cell .markdown }
(A variety of other models is available from other sources - for
example, the [Tensorflow Hub](https://tfhub.dev/).)
:::

::: {.cell .markdown }
I\'m going to use MobileNetV2, which is designed specifically to be
small and fast (so it can run on mobile devices!)

MobileNets come in various sizes controlled by a multiplier for the
depth (number of features), and trained for various sizes of input
images. We will use the 224x224 input image size.
:::

::: {.cell .code }
```python
base_model = tf.keras.applications.MobileNetV2(
  input_shape=INPUT_IMG_SHAPE
)
```
:::

::: {.cell .code }
```python
base_model.summary()
```
:::

::: {.cell .code }
```python
base_probs = base_model.predict(test_sample)
base_probs.shape
```
:::

::: {.cell .code }
```python
url = tf.keras.utils.get_file(
    'ImageNetLabels.txt',
    'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_classes = np.array(open(url).read().splitlines())[1:]
imagenet_classes.shape
```
:::

::: {.cell .markdown }
Let's see what the top 5 predicted classes are for my test image:
:::

::: {.cell .code }
```python
most_likely_classes = np.argsort(base_probs.squeeze())[-5:]
```
:::

::: {.cell .code }
```python
plt.figure(figsize=(10,4));

plt.subplot(1,2,1)
plt.imshow(test_sample.reshape(INPUT_IMG_SIZE, INPUT_IMG_SIZE, 3));

plt.subplot(1,2,2)
p = sns.barplot(x=imagenet_classes[most_likely_classes],y=base_probs.squeeze()[most_likely_classes]);
plt.ylabel("Probability");
p.set_xticklabels(p.get_xticklabels(), rotation=45);
```
:::

::: {.cell .markdown }
MobileNetV2 is trained on a specific task: classifying the images in the
ImageNet dataset by selecting the most appropriate of 1000 class labels.

It is not trained for our specific task: classifying an image of a hand
as rock, paper, or scissors.
:::

::: {.cell .markdown }
## Background: fine-tuning a model
:::

::: {.cell .markdown }
A typical convolutional neural network looks something like this:
:::

::: {.cell .markdown }
![Image via
[PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet)](https://raw.githubusercontent.com/LongerVision/Resource/master/AI/Visualization/PlotNeuralNet/vgg16.png)
:::

::: {.cell .markdown }
We have a sequence of convolutional layers followed by pooling layers.
These layers are *feature extractors* that "learn" key features of our
input images.

Then, we have one or more fully connected layers followed by a fully
connected layer with a softmax activation function. This part of the
network is for *classification*.
:::

::: {.cell .markdown }
The key idea behind transfer learning is that the *feature extractor*
part of the network can be re-used across different tasks and different
domains.
:::

::: {.cell .markdown }
This is especially useful when we don't have a lot of task-specific
data. We can get a pre-trained feature extractor trained on a lot of
data from another task, then train the classifier on task-specific data.
:::

::: {.cell .markdown }
The general process is:

-   Get a pre-trained model, without the classification layer.
-   Freeze the base model.
-   Add a classification layer.
-   Train the model (only the weights in your classification layer will
    be updated).
-   (Optional) Un-freeze some of the last layers in your base model.
-   (Optional) Train the model again, with a smaller learning rate.
:::

::: {.cell .markdown }
## Train our own classification head
:::

::: {.cell .markdown }
This time, we will get the MobileNetV2 model *without* the fully
connected layer at the top of the network.
:::

::: {.cell .code }
```python
import tensorflow.keras.backend as K
K.clear_session()
```
:::

::: {.cell .code }
```python
base_model = tf.keras.applications.MobileNetV2(
  input_shape=INPUT_IMG_SHAPE,
  include_top=False, 
  pooling='avg'
)
```
:::

::: {.cell .code }
```python
base_model.summary()
```
:::

::: {.cell .markdown }
Then, we will *freeze* the model. We\'re not going to train the
MobileNetV2 part of the model, we\'re just going to use it to extract
features from the images.
:::

::: {.cell .code }
```python
base_model.trainable = False
```
:::

::: {.cell .markdown }
We'l make a *new* model out of the "headless" already-fitted
MobileNetV2, with a brand-new, totally untrained classification head on
top:
:::

::: {.cell .code }
```python
model = tf.keras.models.Sequential()

model.add(base_model)
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(
    units=3,
    activation=tf.keras.activations.softmax
))
```
:::

::: {.cell .code }
```python
model.summary()
```
:::

::: {.cell .markdown }
We'l compile the model:
:::

::: {.cell .code }
```python
opt = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(
    optimizer=opt,
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=['accuracy']
)
```
:::

::: {.cell .markdown }
Also, we'l use data augmentation:
:::

::: {.cell .code }
```python
BATCH_SIZE=256
```
:::

::: {.cell .code }
```python
from keras.preprocessing.image import ImageDataGenerator

train_gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                         height_shift_range=0.08, zoom_range=0.08)
train_generator = train_gen.flow(X_train, y_train, batch_size=BATCH_SIZE)

val_gen = ImageDataGenerator()
val_generator = val_gen.flow(X_test, y_test, batch_size=BATCH_SIZE)
```
:::

::: {.cell .markdown }
Now we can start training our model. Remember, we are *only* updating
the weights in the classification head.
:::

::: {.cell .code }
```python
n_epochs = 20

hist = model.fit(
    train_generator, 
    epochs=n_epochs,
    steps_per_epoch=X_train.shape[0]//BATCH_SIZE,
    validation_data=val_generator, 
    validation_steps=X_test.shape[0]//BATCH_SIZE
)
```
:::

::: {.cell .code }
```python
loss = hist.history['loss']
val_loss = hist.history['val_loss']

accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']

plt.figure(figsize=(14, 4))

plt.subplot(1, 2, 1)
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(loss, label='Training set')
plt.plot(val_loss, label='Test set', linestyle='--')
plt.legend()

plt.subplot(1, 2, 2)
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(accuracy, label='Training set')
plt.plot(val_accuracy, label='Test set', linestyle='--')
plt.legend()

plt.show()
```
:::

::: {.cell .markdown }
## Fine-tune model
:::

::: {.cell .markdown }
We have fitted our own classification head, but there\'s one more step
we can attempt to customize the model for our particular application.

We are going to "un-freeze" the later parts of the model, and train it
for a few more epochs on our data, so that the high-level features are
better suited for our specific classification task.
:::

::: {.cell .code }
```python
base_model.trainable = True
```
:::

::: {.cell .code }
```python
len(base_model.layers)
```
:::

::: {.cell .markdown }
Note that we are *not* creating a new model. We\'re just going to
continue training the model we already started training.
:::

::: {.cell .code }
```python
fine_tune_at = 149

# freeze first layers
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable =  False
    
# use a smaller training rate for fine-tuning
opt = tf.keras.optimizers.Adam(learning_rate=0.00001)
model.compile(
    optimizer = opt,
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=['accuracy']
)

model.summary()
```
:::

::: {.cell .code }
```python
n_epochs_fine = 20

hist_fine = model.fit( 
    train_generator, 
    epochs=n_epochs + n_epochs_fine,
    initial_epoch=n_epochs,  
    steps_per_epoch=X_train.shape[0]//BATCH_SIZE,
    validation_data=val_generator, 
    validation_steps=X_test.shape[0]//BATCH_SIZE
)
```
:::

::: {.cell .code }
```python
loss = hist.history['loss'] + hist_fine.history['loss']
val_loss = hist.history['val_loss'] + hist_fine.history['val_loss']

accuracy = hist.history['accuracy'] + hist_fine.history['accuracy']
val_accuracy = hist.history['val_accuracy'] + hist_fine.history['val_accuracy']

plt.figure(figsize=(14, 4))

plt.subplot(1, 2, 1)
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(loss, label='Training set')
plt.plot(val_loss, label='Test set', linestyle='--')
plt.plot([n_epochs, n_epochs], plt.ylim(),label='Fine Tuning',linestyle='dotted')
plt.legend()

plt.subplot(1, 2, 2)
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(accuracy, label='Training set')
plt.plot(val_accuracy, label='Test set', linestyle='dotted')
plt.plot([n_epochs, n_epochs], plt.ylim(), label='Fine Tuning', linestyle='--')
plt.legend()

plt.show()
```
:::

::: {.cell .markdown }
## Classify custom test sample
:::

::: {.cell .code }
```python
test_probs = model.predict(test_sample)
```
:::

::: {.cell .code }
```python
plt.figure(figsize=(10,4));

plt.subplot(1,2,1)
plt.imshow(test_sample.reshape(INPUT_IMG_SIZE, INPUT_IMG_SIZE, 3));

plt.subplot(1,2,2)
p = sns.barplot(x=classes,y=test_probs.squeeze());
plt.ylabel("Probability");
```
:::

::: {.cell .markdown }
## Some comments
:::

::: {.cell .markdown }
In practice, for most machine learning problems, you wouldn't design or
train a convolutional neural network from scratch - you would use an
existing model that suits your needs (does well on ImageNet, size is
right) and fine-tune it on your own data.
:::

::: {.cell .markdown }
Transfer learning isn't only for image classification.

There are many problems that can be solved by taking a VERY LARGE
task-generic "feature detection" model trained on a LOT of data, and
fine-tuning it on a small custom dataset.
:::

::: {.cell .markdown }
For example, consider [AI Dungeon](https://play.aidungeon.io/), a game
in the style of classic text-based adventure games.

It was trained by fine-tuning a version of GPT-2. GPT-2 is a language
model with 1.5 billion parameters, trained on a dataset of 8 million web
pages, with the objective of predicting the next word in a sequence.

The creator of AI Dungeon fine-tuned GTP-2 using story-games scraped
from [ChooseYourStory.com](https://chooseyourstory.com/Stories/).

With so much data, the model doesn't only learn about language and
language features - it learns about the world described by all that
text! Since the game is based on a fine-tuned version of that model, it
also knows a lot about the world.
:::
