---
title: 'Convolutional neural networks'
author: 'Fraida Fund'
jupyter:
  accelerator: GPU
  colab:
    toc_visible: true
  kernelspec:
    display_name: Python 3
    name: python3
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .markdown }
# Convolutional neural networks

*Fraida Fund*
:::

::: {.cell .markdown }
In this notebook, we will find out makes convolutional neural networks
so powerful for computer vision applications!

We will use three varieties of neural networks to classify our own
handwritten digits.
:::

::: {.cell .markdown }
Note: for faster training, use Runtime \> Change Runtime Type to run
this notebook on a GPU.
:::

::: {.cell .code }
```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
```
:::

::: {.cell .markdown }
Check if you are using a GPU -
:::

::: {.cell .code }
```python
if torch.cuda.is_available():
  print(torch.cuda.get_device_name(0))
else:
  print("No GPU available.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
:::

::: {.cell .markdown }
We set the `device` above - throughout this notebook, we will move models and data to this `device` for computation.
:::


::: {.cell .markdown }
## Train a fully connected neural network on MNIST
:::

::: {.cell .markdown }
First, we will train a simple neural network. 

:::

::: {.cell .markdown}

Let's get the MNIST dataset from Pytorch's `datasets` module. We will also define some "transforms" that will be applied to each batch of data - in this case, we will make it into a "tensor" and normalize it using the mean and standard deviation of MNIST pixels.

:::


::: {.cell .code }
```python
basic_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.1307, std=0.3081)
])

# get datasets with basic transform
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=basic_transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=basic_transform)
```
:::

::: {.cell .markdown}

Then we will prepare data loaders. We will use early stopping, which is a kind of model selection, so we will need a separate validation set to be split out of the training set for that.

:::


::: {.cell .code }
```python

val_size = len(train_dataset) // 6
train_size = len(train_dataset) - val_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)
```
:::


::: {.cell .markdown }

Let's look at one data sample - specifically, its shape: it is a 3D volume with dimensions [C, H, W] (number of channels, height, and width). 

:::

::: {.cell .code }
```python
train_dataset[0][0].shape
```
:::


::: {.cell .markdown }

For a fully connected neural network, we will need to "flatten" the data from its current 3D 1x28x28 shape to a 1D 784 shape, using a `Flatten` layer right after the input.  

Then, we will have:

-   One hidden layer with $N_H=512$ units, with ReLu activation.
-   One output layer with $N_O=10$ units, one for each of the 10 possible classes. 
-   The output will be logits (pre-softmax), and then we can use `argmax` to find the most probable class.

:::


::: {.cell .markdown }
With that in mind, we can define our fully connected neural network.
:::

::: {.cell .code }
```python

class FCNet(nn.Module):
    def __init__(self, input_dim=28*28, hidden_dim=512, output_dim=10):
        super(FCNet, self).__init__()
        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.flatten(x)           # [B, 1, 28, 28] → [B, 784]
        x = F.relu(self.hidden(x))   
        x = self.output(x)           
        return x                     
```
:::

::: {.cell .code }
```python
model_fc = FCNet().to(device)
```
:::

::: {.cell .markdown }
To train the network, we have to select an optimizer and a loss
function. Since this is a multi-class classification problem with logit output,
we use `CrossEntropyLoss`. We use the Adam optimizer
for our gradient descent.

:::

::: {.cell .code }
```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_fc.parameters(), lr=0.005)

```
:::

::: {.cell .markdown }
Finally, we are ready to train our network. 

We wil specify the number of epochs. 
But we will also configure the training process to stop before the max number of
epochs, if no we hit a "peak" in the validation set accuracy and then fail to improve on it for
"patience" consecutive epochs. 

We will also save the model weights each time we achieve a new "peak" performance on the validation set,
and at the end, we will restore the weights that had the best
performance on the validation set.
:::

::: {.cell .code }
```python
best_val_acc = 0
best_model_state = None
patience = 5
counter = 0

train_loss_history = []
val_acc_history = []

for epoch in range(100):

    # Train for one epoch
    model_fc.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model_fc(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Evaluate on validation set at the end of the epoch
    model_fc.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_fc(images)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    train_loss_history.append(running_loss)
    val_acc_history.append(val_acc)

    # Check if early stopping should be triggered
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model_fc.state_dict(), "best_fc_model.pt")  # save model weights to file if it's the best so far
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Epoch {epoch+1:03d} - Loss: {running_loss:.4f} - Val Acc: {val_acc:.4f} - Early Stopping Triggered")
            break
            
    print(f"Epoch {epoch+1:03d} - Loss: {running_loss:.4f} - Val Acc: {val_acc:.4f} - Patience Counter {counter}")

# Restore whatever "best model" we found
model_fc.load_state_dict(torch.load("best_fc_model.pt"))
print(f"Best validation accuracy: {best_val_acc:.4f}")
```
:::

::: {.cell .markdown }
Now we can make predictions with our fitted model, and compute accuracy on the test set:
:::

::: {.cell .code }
```python
model_fc.eval()  # Set model to evaluation mode
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model_fc (images)
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Compute test accuracy
test_acc = (np.array(y_true) == np.array(y_pred)).mean()
print(f"Test Accuracy: {test_acc:.4f}")
```
:::


::: {.cell .markdown }

Our neural network does OK, although not great! Currently, the state of the art
(best result) on the MNIST dataset is 0.21% classification error - you
can see some of the best-performing methods at [this
link](https://benchmarks.ai/mnist).

Looking at some of the samples that are misclassified by
our fully connected network, we can see that many of these samples are difficult for
humans to classify as well. (Some may even be labeled incorrectly!)
The human error rate on MNIST is likely to be about 0.2-0.3%.

:::

::: {.cell .code }
```python
y_pred = np.array(y_pred).flatten()
y_true = np.array(y_true).flatten()

# find misclassified samples
mis_idx = np.where(y_pred != y_true)[0]
num_samples = min(10, len(mis_idx))  
chosen_idx = np.random.choice(mis_idx, num_samples, replace=False)

plt.figure(figsize=(num_samples * 1.25, 2))
for i, idx in enumerate(chosen_idx):
    image, _ = test_dataset[idx]
    plt.subplot(1, num_samples, i + 1)
    sns.heatmap(image.squeeze(), cmap=plt.cm.gray, cbar=False,
                xticklabels=False, yticklabels=False)
    plt.axis('off')
    plt.title(f"Idx {idx}\nTrue: {y_true[idx]}\nPred: {y_pred[idx]}")
plt.tight_layout()
plt.show()
```
:::

::: {.cell .markdown }
## Try our fully connected neural network on our own test sample
:::

::: {.cell .markdown }
Now, let's try to classify our own test sample (as in a previous
notebook, when we did this for a logistic regression model).

On a plain white piece of paper, in a black or other dark-colored pen,
write a digit of your choice from 0 to 9. Take a photo of your
handwritten digit.

Edit your photo (crop, rotate as needed), using a photo editor of your
choice (I used Google Photos), so that your photo is approximately
square, and includes only the digit and the white background. Upload
your image here.
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
 
filename = 'input.png'
 
image = Image.open(filename)
p = plt.imshow(np.asarray(image), cmap=plt.cm.gray,);
p = plt.title('Shape: ' + str(np.asarray(image).shape))
```
:::

::: {.cell .code }
```python
# convert to grayscale image - 'L' format means each pixel is 
# represented by a single value from 0 to 255
image_bw = image.convert('L')
p = plt.imshow(np.asarray(image_bw), cmap=plt.cm.gray,);
p = plt.title('Shape: ' + str(np.asarray(image_bw).shape))
```
:::

::: {.cell .code }
```python
# resize image 
image_bw_resized = image_bw.resize((28,28), Image.BICUBIC)
p = plt.imshow(np.asarray(image_bw_resized), cmap=plt.cm.gray,);
p = plt.title('Shape: ' + str(np.asarray(image_bw_resized).shape))
```
:::

::: {.cell .code }
```python
# invert image, to match training data
import PIL.ImageOps    

image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
p = plt.imshow(np.asarray(image_bw_resized_inverted), cmap=plt.cm.gray,);
p = plt.title('Shape: ' + str(np.asarray(image_bw_resized_inverted).shape))
```
:::

::: {.cell .code }
```python
# finally, turn to a numpy array
test_sample = np.array(image_bw_resized_inverted).reshape(1, 28, 28)
p = plt.imshow(np.reshape(test_sample, (28,28)), cmap=plt.cm.gray,);
p = plt.title('Shape: ' + str(test_sample.shape))
```
:::

::: {.cell .markdown }
Now we can predict the class of this sample:
:::

::: {.cell .code }
```python
input_tensor = basic_transform(test_sample.squeeze())  # shape: [1, 28, 28] → [28, 28] → tensor [1, 28, 28]
input_tensor = input_tensor.unsqueeze(0).to(device)  # shape: [1, 1, 28, 28]
input_tensor.shape
```
:::

::: {.cell .code }
```python
model_fc.eval()
with torch.no_grad():
    output = model_fc(input_tensor)
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()

print("Predicted class:", predicted_class)
```
:::

::: {.cell .code }
```python
sns.barplot(x=np.arange(0,10), y=probabilities.cpu().squeeze());
plt.ylabel("Probability");
plt.xlabel("Class");
```
:::

::: {.cell .markdown }
### Things to try

-   What if we use a test sample where the image is not so well
    centered?
:::

::: {.cell .markdown }
## Background: Convolutional neural networks
:::

::: {.cell .markdown }
The fully connected neural network was OK, but for images, there are
important reasons why we will often prefer a convolutional neural
network instead:

-   Dimension - images can have a huge number of pixels, and for image
    classification problems, we can also have a very large number of
    possible classes. A deep, fully connected network for these problems
    will have a *lot* of weights to learn.
-   Images (and videos!) have a structure that is wasted on the fully
    connected network.
-   Relevant features may be anywhere in the image.
:::

::: {.cell .markdown }
The key idea behind convolutional neural networks is that a "neuron"
is connected to a small part of image at a time (locally connected).

By having multiple locally connected neurons covering the entire image,
we effectively "scan" the image.
:::

::: {.cell .markdown }
What does convolution do? Let's look at a visual example.
:::

::: {.cell .markdown }
This is a horizontal Sobel filter, which detects horizontal edges.
:::

::: {.cell .code }
```python
horizontal_sobel = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
plt.imshow(horizontal_sobel, cmap='RdBu');
```
:::

::: {.cell .markdown }
This is an image of random noise:
:::

::: {.cell .code }
```python
img = np.random.uniform(0,1,size=(10,10))
plt.imshow(img, cmap='gray');
```
:::

::: {.cell .markdown }
The convolution of the Sobel filter and the random image doesn't pick
up anything interesting:
:::

::: {.cell .code }
```python
from scipy import signal
img_conv = signal.correlate2d(img, horizontal_sobel, mode='same')
plt.imshow(img_conv, cmap='gray');
```
:::

::: {.cell .markdown }
What about the convolution of the Sobel filter and this digit?
:::

::: {.cell .code }
```python
img_index = 3675
img = test_dataset[img_index][0].squeeze().cpu().numpy()
plt.imshow(img.reshape(28,28), cmap='gray');
```
:::

::: {.cell .code }
```python
img_conv = signal.correlate2d(img.reshape(28,28), horizontal_sobel, mode='same')
plt.imshow(img_conv, cmap='gray');
```
:::

::: {.cell .markdown }
This is a vertical Sobel filter, which detects vertical edges.
:::

::: {.cell .code }
```python
vertical_sobel =  np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
plt.imshow(vertical_sobel, cmap='RdBu');
```
:::

::: {.cell .markdown }
Look what it finds in the digit -
:::

::: {.cell .code }
```python
img_conv = signal.correlate2d(img.reshape(28,28), vertical_sobel, mode='same')
plt.imshow(img_conv, cmap='gray');
```
:::

::: {.cell .markdown }
A convolutional layer is like an array of these filters - each one
"sweeps" the image and looks for a different high-level "feature".
:::

::: {.cell .markdown }
*Attribution: this example is based on a post by [Victor
Zhou](https://victorzhou.com/blog/intro-to-cnns-part-1/).*
:::

::: {.cell .markdown }
You can see a great interactive demo of the Sobel filters in [this
tutorial on edge
detection](https://cse442-17f.github.io/Sobel-Laplacian-and-Canny-Edge-Detection-Algorithms/).
:::

::: {.cell .markdown }
## Train a convolutional neural network on MNIST
:::

::: {.cell .markdown }
In this next section, we will train a convolutional neural network.

We have explicitly defined it in two parts, a feature extraction part and a classification part, for clarify.

Also, we will try to improve performance using the following techniques:

-   **Dropout layers**: Because deep networks can be prone to
    overfitting, we will also add *dropout* layers to our network
    architecture. In each training stage, a dropout layer will "zero"
    a random selection of outputs (just for that stage). You can read
    more about this technique in [this
    paper](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf).
-   **Batch normalization**: This technique re-scales and centers the
    data in the mini-batch when applied between layers.
:::



::: {.cell .markdown }
Note that this time we do *not* `Flatten` the data at the input. The input is processed directly as a 
3D volume through a sequence of `Conv2D`, `BatchNormalization`, `Activation`, `MaxPooling2D`, and 
`Dropout` layers, and finally `Flatten` and `Dense` layers.
:::

::: {.cell .code }
```python
class ConvNet(nn.Module):
    def __init__(self, n_classes=10):
        super(ConvNet, self).__init__()

        self.features = nn.Sequential(
            # Conv Layer 1
            nn.Conv2d(1, 32, kernel_size=3),         # input: [B, 1, 28, 28] → [B, 32, 26, 26]
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Conv Layer 2
            nn.Conv2d(32, 32, kernel_size=3),        # → [B, 32, 24, 24]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),             # → [B, 32, 12, 12]

            # Conv Layer 3
            nn.Conv2d(32, 64, kernel_size=3),        # → [B, 64, 10, 10]
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Conv Layer 4
            nn.Conv2d(64, 64, kernel_size=3),        # → [B, 64, 8, 8]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),             # → [B, 64, 4, 4]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                            # → [B, 64*4*4 = 1024]
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, n_classes)                # logits output
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```
:::


::: {.cell .code }
```python
model_conv = ConvNet().to(device)
```
:::

::: {.cell .code }
```python
model_conv
```
:::



::: {.cell .code }
```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_conv.parameters(), lr=0.005)

```
:::

::: {.cell .markdown }

As before, we will train the model with early stopping for up to 100 epochs.

:::

::: {.cell .code }
```python
best_val_acc = 0
best_model_state = None
patience = 5
counter = 0

train_loss_history = []
val_acc_history = []

for epoch in range(100):

    # Train for one epoch
    model_conv.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model_conv(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Evaluate on validation set at the end of the epoch
    model_conv.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_conv(images)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    train_loss_history.append(running_loss)
    val_acc_history.append(val_acc)

    # Check if early stopping should be triggered
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model_conv.state_dict(), "best_conv_model.pt")  # save model weights to file if it's the best so far
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Epoch {epoch+1:03d} - Loss: {running_loss:.4f} - Val Acc: {val_acc:.4f} - Early Stopping Triggered")
            break
            
    print(f"Epoch {epoch+1:03d} - Loss: {running_loss:.4f} - Val Acc: {val_acc:.4f} - Patience Counter {counter}")

# Restore whatever "best model" we found
model_conv.load_state_dict(torch.load("best_conv_model.pt"))
print(f"Best validation accuracy: {best_val_acc:.4f}")
```
:::



::: {.cell .code }
```python
model_conv.eval()  # Set model to evaluation mode
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model_conv (images)
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Compute test accuracy
test_acc = (np.array(y_true) == np.array(y_pred)).mean()
print(f"Test Accuracy: {test_acc:.4f}")
```
:::

::: {.cell .markdown }

This is a much better result!

:::

::: {.cell .markdown }
These are some of the samples that are misclassified:
:::

::: {.cell .code }
```python
y_pred = np.array(y_pred).flatten()
y_true = np.array(y_true).flatten()

# find misclassified samples
mis_idx = np.where(y_pred != y_true)[0]
num_samples = min(10, len(mis_idx))  
chosen_idx = np.random.choice(mis_idx, num_samples, replace=False)

plt.figure(figsize=(num_samples * 1.25, 2))
for i, idx in enumerate(chosen_idx):
    image, _ = test_dataset[idx]
    plt.subplot(1, num_samples, i + 1)
    sns.heatmap(image.squeeze(), cmap=plt.cm.gray, cbar=False,
                xticklabels=False, yticklabels=False)
    plt.axis('off')
    plt.title(f"Idx {idx}\nTrue: {y_true[idx]}\nPred: {y_pred[idx]}")
plt.tight_layout()
plt.show()
```
:::



::: {.cell .markdown }
## Try our convolutional neural network on our own test sample
:::

::: {.cell .markdown }
We can use this convolutional neural network to predict the class of the
test sample we uploaded previously.
:::

::: {.cell .code }
```python
plt.imshow(test_sample.reshape(28, 28), cmap='gray');
```
:::

::: {.cell .code }
```python
model_conv.eval()
with torch.no_grad():
    output = model_conv(input_tensor)
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()

print("Predicted class:", predicted_class)
```
:::

::: {.cell .code }
```python
sns.barplot(x=np.arange(0,10), y=probabilities.cpu().squeeze());
plt.ylabel("Probability");
plt.xlabel("Class");
```
:::



::: {.cell .markdown }
## Looking at output of convolutional layers

Because deep learning is so complex, it can be difficult to understand
why it makes the decisions it does. One way to better understand the
behavior of a neural network is to visualize the output of each layer
for a given input.
:::

::: {.cell .code }
```python
#@title ### Visualization magic

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact, IntSlider
from IPython.display import display

activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach().cpu()
    return hook

layer_names = []
for name, layer in model_conv.named_modules():
    if isinstance(layer, nn.Conv2d):
        pre_name = f"{name}_pre"
        layer.register_forward_hook(get_activation(pre_name))
        layer_names.append(pre_name)
    if isinstance(layer, nn.ReLU):
        layer.register_forward_hook(get_activation(name))
        layer_names.append(name)

def visualize_layer(test_idx=0, layer_idx=0):
    model_conv.eval()
    img, _ = test_dataset[test_idx]
    x_tensor = img.unsqueeze(0).to(device)
    with torch.no_grad():
        _ = model_conv(x_tensor)
    name = layer_names[layer_idx]
    fmap = activations[name].squeeze(0)
    n_channels = fmap.shape[0]
    if fmap.ndim == 1 or (fmap.shape[1:] == torch.Size([1, 1])):
        fmap_flat = fmap.view(-1).numpy()
        plt.figure(figsize=(len(fmap_flat) * 0.25, 2))
        plt.imshow(fmap_flat[np.newaxis, :], cmap='gray', aspect='auto')
        plt.title(f"{name} (Vector Output)")
        plt.yticks([]); plt.xticks(range(len(fmap_flat)))
        plt.show()
        return
    n_cols = int(np.ceil(np.sqrt(n_channels)))
    n_rows = int(np.ceil(n_channels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    fig.suptitle(name)
    for i in range(n_channels):
        ax = axes.flat[i]
        ax.imshow(fmap[i], cmap='gray')
        ax.set_title(f"{i}", fontsize=8)
        ax.axis('off')
    for i in range(n_channels, len(axes.flat)):
        axes.flat[i].axis('off')
    plt.tight_layout()
    plt.show()

widget = interact(
    visualize_layer,
    test_idx=IntSlider(0, 0, len(test_dataset) - 1, step=1, description='Test Index'),
    layer_idx=IntSlider(0, 0, len(layer_names) - 1, step=1, description='Layer Index')
)

```
:::

::: {.cell .markdown }
Generally, the convolutional layers close to the input capture small
details, while those close to the output of the model capture more
general features that are less sensitive to local variations in the
input image. We can see this characteristic in the visualizations above.
:::


::: {.cell .markdown }
## Saving and restoring a model
:::

::: {.cell .markdown }
Since this model took a long time to train, it may be useful to save the
results, so that we can re-use the model later without having to
re-train. We can save the weights of a model using `torch.save`:
:::

::: {.cell .code }
```python
torch.save(model_conv.state_dict(), "saved_model.pt") 
```
:::

::: {.cell .markdown }
Now, if you click on the folder icon in the menu on the left side of the
Colab window, you can see this file in your workspace. You can download
the file for later use.

To use the model again in the future, you can load it using
`load_model`, then use it to make predictions without having to train
it. 

Here we define a brand-new instance of a `ConvNet`, and then load the already-trained weights:
:::


::: {.cell .code }
```python
model_conv_new = ConvNet().to(device)
model_conv_new.load_state_dict(torch.load("saved_model.pt", map_location=device))
```
:::

::: {.cell .markdown }
## With data augmentation
:::

::: {.cell .markdown }
We can try one more way to improve the model performance:

-   **Data augmentation**: To supply more training samples, we can
    provide slightly modified versions of training samples - for
    example, samples with a small rotation applied - on which to train
    the model.

We will implement this by modifying the `transform` that is passed to the dataset:
:::

::: {.cell .code }
```python
aug_transform = transforms.Compose([
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.1, contrast=0.1)
    ], p=1.0),
    transforms.RandomAffine(
        degrees=10,                # rotate between [-10°, +10°]
        translate=(0.1, 0.1),      # shift up to 10% in x and y
        scale=(0.95, 1.05)         # zoom in/out slightly
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# get datasets with augmented transform
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=aug_transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=aug_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

```
:::


::: {.cell .markdown }

Then, we will train a `ConvNet` exactly as before, but using this augmented data.

:::

::: {.cell .code }
```python
model_aug = ConvNet().to(device)
```
:::


::: {.cell .code }
```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_aug.parameters(), lr=0.005)

```
:::


::: {.cell .code }
```python
best_val_acc = 0
best_model_state = None
patience = 5
counter = 0

train_loss_history = []
val_acc_history = []

for epoch in range(100):

    # Train for one epoch
    model_aug.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model_aug(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Evaluate on validation set at the end of the epoch
    model_aug.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_aug(images)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    train_loss_history.append(running_loss)
    val_acc_history.append(val_acc)

    # Check if early stopping should be triggered
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model_aug.state_dict(), "best_aug_model.pt")  # save model weights to file if it's the best so far
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Epoch {epoch+1:03d} - Loss: {running_loss:.4f} - Val Acc: {val_acc:.4f} - Early Stopping Triggered")
            break
            
    print(f"Epoch {epoch+1:03d} - Loss: {running_loss:.4f} - Val Acc: {val_acc:.4f} - Patience Counter {counter}")

# Restore whatever "best model" we found
model_aug.load_state_dict(torch.load("best_aug_model.pt"))
print(f"Best validation accuracy: {best_val_acc:.4f}")
```
:::



::: {.cell .code }
```python
model_aug.eval()  # Set model to evaluation mode
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model_aug (images)
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Compute test accuracy
test_acc = (np.array(y_true) == np.array(y_pred)).mean()
print(f"Test Accuracy: {test_acc:.4f}")
```
:::

::: {.cell .markdown }
These are some of the samples that are misclassified:
:::


::: {.cell .code }
```python
y_pred = np.array(y_pred).flatten()
y_true = np.array(y_true).flatten()

# find misclassified samples
mis_idx = np.where(y_pred != y_true)[0]
num_samples = min(10, len(mis_idx))  
chosen_idx = np.random.choice(mis_idx, num_samples, replace=False)

plt.figure(figsize=(num_samples * 1.25, 2))
for i, idx in enumerate(chosen_idx):
    image, _ = test_dataset[idx]
    plt.subplot(1, num_samples, i + 1)
    sns.heatmap(image.squeeze(), cmap=plt.cm.gray, cbar=False,
                xticklabels=False, yticklabels=False)
    plt.axis('off')
    plt.title(f"Idx {idx}\nTrue: {y_true[idx]}\nPred: {y_pred[idx]}")
plt.tight_layout()
plt.show()
```
:::


::: {.cell .markdown }

and here is our test sample:

:::


::: {.cell .code }
```python
plt.imshow(test_sample.reshape(28, 28), cmap='gray');
```
:::

::: {.cell .code }
```python
model_aug.eval()
with torch.no_grad():
    output = model_aug(input_tensor)
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()

print("Predicted class:", predicted_class)
```
:::

::: {.cell .code }
```python
sns.barplot(x=np.arange(0,10), y=probabilities.cpu().squeeze());
plt.ylabel("Probability");
plt.xlabel("Class");
```
:::



::: {.cell .markdown }
## Try more of your own test samples!
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
 
filename = 'input2.png'
 
image = Image.open(filename)
image_bw = image.convert('L')
image_bw_resized = image_bw.resize((28,28), Image.BICUBIC)
image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
test_sample = np.array(image_bw_resized_inverted).reshape(1, 28, 28)
p = plt.imshow(np.reshape(test_sample, (28,28)), cmap=plt.cm.gray,);
p = plt.title('Shape: ' + str(test_sample.shape))
```
:::

::: {.cell .code }
```python
input_tensor = basic_transform(test_sample.squeeze())  # shape: [1, 28, 28] → [28, 28] → tensor [1, 28, 28]
input_tensor = input_tensor.unsqueeze(0).to(device)  # shape: [1, 1, 28, 28]
input_tensor.shape
```
:::

::: {.cell .code }
```python
model_fc.eval()
with torch.no_grad():
    output = model_fc(input_tensor)
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()

sns.barplot(x=np.arange(0,10), y=probabilities.cpu().squeeze());
plt.ylabel("Probability");
plt.xlabel("Class");
plt.title("Fully connected network\nPredicted class: %d" % int(predicted_class));
```
:::


::: {.cell .code }
```python
model_conv.eval()
with torch.no_grad():
    output = model_conv(input_tensor)
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()

sns.barplot(x=np.arange(0,10), y=probabilities.cpu().squeeze());
plt.ylabel("Probability");
plt.xlabel("Class");
plt.title("Convolutional network\nPredicted class: %d" % int(predicted_class));

```
:::

::: {.cell .code }
```python
model_aug.eval()
with torch.no_grad():
    output = model_aug(input_tensor)
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()

sns.barplot(x=np.arange(0,10), y=probabilities.cpu().squeeze());
plt.ylabel("Probability");
plt.xlabel("Class");
plt.title("Convolutional network with data augmentation\nPredicted class: %d" % int(predicted_class));

```
:::


::: {.cell .markdown }
## More things to try

-   This notebook runs using a free GPU on Colab! Try changing the
    runtime to CPU: Runtime \> Change Runtime Type and change Hardware
    Accelerator to CPU. Then run the notebook again. How much speedup
    did you get with the GPU, relative to CPU?
:::
