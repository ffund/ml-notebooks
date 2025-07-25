---
title: 'Fine tune a network on rock paper scissors data'
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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

```
:::

::: {.cell .markdown }
## Import data
:::

::: {.cell .markdown }
In the cells that follow, we'll get the "rock paper scissors" data, plot
a few examples, and also prepare a preprocessing function (which we won't apply yet). The preprocessing transforms will:

* resize each sample to 224x224 (this is a typical input size for pretrained models trained on ImageNet)
* and normalize input using the mean and standard deviation of ImageNet

:::

::: {.cell .code }
```python
import os, urllib.request, zipfile

urls = {
    'train': "https://storage.googleapis.com/download.tensorflow.org/data/rps.zip",
    'test':  "https://storage.googleapis.com/download.tensorflow.org/data/rps-test-set.zip"
}
data_dir = "./data/rps"

os.makedirs(data_dir, exist_ok=True)

for split, url in urls.items():
    zip_path = os.path.join(data_dir, f"{split}.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading {split} set...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(os.path.join(data_dir, split))
        print(f"{split} set extracted.")
```
:::

::: {.cell .code }
```python
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

```
:::

::: {.cell .code }
```python
train_root = os.path.join(data_dir, 'train', 'rps')
test_root  = os.path.join(data_dir, 'test', 'rps-test-set')

train_dataset = datasets.ImageFolder(root=train_root, transform=transform)
test_dataset  = datasets.ImageFolder(root=test_root, transform=transform)
class_names = train_dataset.classes

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=32)
```
:::

::: {.cell .code }
```python
n = 5
mean = torch.tensor(imagenet_mean).view(3, 1, 1)
std = torch.tensor(imagenet_std).view(3, 1, 1)

idxs = np.random.choice(len(train_dataset), n, replace=False)

plt.figure(figsize=(n * 2, 2))
for i, idx in enumerate(idxs):
    img, label = train_dataset[idx]
    img = img * std + mean  # de-normalize
    img = img.permute(1, 2, 0).clamp(0, 1).numpy()
    plt.subplot(1, n, i + 1)
    plt.imshow(img)
    plt.title(class_names[label])
    plt.axis('off')
plt.tight_layout()
plt.show()
```
:::


::: {.cell .markdown }
## Classify with a ResNet
:::

::: {.cell .markdown }

[torchvision models](https://docs.pytorch.org/vision/0.9/models.html) image models that have been 
pre-trained on ImageNet. You can download their saved weights, and use in your own code.

We are going to use a pre-trained ResNet model

:::



::: {.cell .code }
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

```
:::

::: {.cell .code }
```python
base_model = models.resnet18(weights='ResNet18_Weights.DEFAULT').to(device)

```
:::


::: {.cell .code }
```python
base_model
```
:::


::: {.cell .code }
```python
# Get class labels
labels_url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
labels_path = 'imagenet_classes.txt'
if not os.path.exists(labels_path):
    urllib.request.urlretrieve(labels_url, labels_path)

with open(labels_path) as f:
    class_labels = [line.strip() for line in f]

```
:::

::: {.cell .markdown }
Let's see what the top 5 predicted classes are for my test image:
:::

::: {.cell .code }
```python
base_model.eval()
with torch.no_grad():
    output = base_model(test_tensor.to(device))              # [1, 1000]
    probs = F.softmax(output, dim=1)
    top_prob, top_idx = probs[0].topk(5)
```
:::

::: {.cell .code }
```python
top_classes = [class_labels[i] for i in top_idx.tolist()]
top_probs = top_prob.cpu().numpy()

plt.figure(figsize=(6, 4))
ax = sns.barplot(x=top_classes, y=top_probs)
ax.set_ylabel("Probability")
ax.set_title("Top-5 Predictions")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.tight_layout()
plt.show()
```
:::

::: {.cell .markdown }
The base model is trained on a specific task: classifying the images in the
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

This time, we will 

* get the base model, 
* freeze the weights in the feature extraction part, 
* and put a brand-new, totally untrained classification head on top.

:::


::: {.cell .code }
```python
transfer_model = models.resnet18(weights='ResNet18_Weights.DEFAULT')

# Freeze all parameters
for param in transfer_model.parameters():
    param.requires_grad = False

# Replace the classification head at the end
num_ftrs = transfer_model.fc.in_features
transfer_model.fc = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(num_ftrs, 3)  # 3 classes: rock, paper, scissors
)
```
:::

::: {.cell .code }
```python
transfer_model = transfer_model.to(device)

```
:::

::: {.cell .code }
```python
transfer_model
```
:::


::: {.cell .code }
```python
total_params = sum(
	param.numel() for param in transfer_model.parameters()
)
trainable_params = sum(
	p.numel() for p in transfer_model.parameters() if p.requires_grad
)

print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")
```
:::


::: {.cell .markdown }
Here, we apply data augmentation and preprocessing to the training data; and just preprocessing to the test data.
:::

::: {.cell .code }
```python
# data augmentation for training set
aug_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

# no augmentation for test set
basic_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

train_dataset = datasets.ImageFolder(root=train_root, transform=aug_transform)
test_dataset  = datasets.ImageFolder(root=test_root,  transform=basic_transform)
class_names   = train_dataset.classes


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64)

```
:::

::: {.cell .markdown }
Now we can start training our model. Remember, we are *only* updating
the weights in the classification head.

Also note that we are reporting loss on the test data in each epoch, but we are
not doing early stopping or otherwise "using" this. If we were using this loss
to make decisions about the training process, we would have to split out a separate 
validation set to avoid data leakage.

:::

::: {.cell .code }
```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(transfer_model.parameters(), lr=1e-4)
```
:::

::: {.cell .code }
```python
num_epochs = 20
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []


for epoch in range(num_epochs):
    # Train on training set
    transfer_model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = transfer_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    train_losses.append(total_loss / total)
    train_accuracies.append(correct / total)

    # Evaluate on test set
    transfer_model.eval()
    test_loss, test_correct, test_total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = transfer_model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            test_correct += (outputs.argmax(1) == labels).sum().item()
            test_total += labels.size(0)

    test_losses.append(test_loss / test_total)
    test_accuracies.append(test_correct / test_total)

    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {train_losses[-1]:.4f}, Acc: {train_accuracies[-1]*100:.2f}% | "
          f"Test Loss: {test_losses[-1]:.4f}, Acc: {test_accuracies[-1]*100:.2f}%")

```
:::

::: {.cell .code }
```python
plt.figure(figsize=(5, 4))
plt.plot(train_losses, marker='o', label='Train Loss')
plt.plot(test_losses, marker='s', label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
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
# Unfreeze the last n_unfreeze layers of the feature extractor
# Unfreeze last residual block (layer4)
for param in transfer_model.layer4.parameters():
    param.requires_grad = True

transfer_model = transfer_model.to(device)
```
:::


::: {.cell .code }
```python
total_params = sum(
	param.numel() for param in transfer_model.parameters()
)
trainable_params = sum(
	p.numel() for p in transfer_model.parameters() if p.requires_grad
)

print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")
```
:::

::: {.cell .markdown }

We will fine-tune these additional parameters using a smaller learning rate:

:::

::: {.cell .code}
```python
optimizer = optim.Adam(transfer_model.parameters(), lr=1e-7)
```
:::

::: {.cell .markdown }
Note that we are *not* creating a new model. We're just going to
continue training the model we already started training.

:::


::: {.cell .code }
```python
num_epochs_fine = 20

for epoch in range(num_epochs + 1, num_epochs + num_epochs_fine + 1):
    # Train on training set
    transfer_model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = transfer_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    train_losses.append(total_loss / total)
    train_accuracies.append(correct / total)

    # Evaluate on test set
    transfer_model.eval()
    test_loss, test_correct, test_total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = transfer_model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            test_correct += (outputs.argmax(1) == labels).sum().item()
            test_total += labels.size(0)

    test_losses.append(test_loss / test_total)
    test_accuracies.append(test_correct / test_total)

    print(f"Epoch {epoch}/{num_epochs + num_epochs_fine} | "
          f"Train Loss: {train_losses[-1]:.4f}, Acc: {train_accuracies[-1]*100:.2f}% | "
          f"Test Loss: {test_losses[-1]:.4f}, Acc: {test_accuracies[-1]*100:.2f}%")
```
:::

::: {.cell .code }
```python
plt.figure(figsize=(8, 5))
plt.plot(train_losses, marker='o', label='Train Loss')
plt.plot(test_losses, marker='s', label='Test Loss')

# vertical dashed line to mark fine-tuning start
plt.axvline(x=num_epochs - 1, color='gray', linestyle='--', label='Fine-tuning start')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```
:::

::: {.cell .markdown }
## Classify custom test sample
:::

::: {.cell .markdown }
 
Let us also upload a personal example, in PNG format.

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
test_tensor = transform(image).unsqueeze(0)  # shape: [1, 3, 224, 224]
```
:::

::: {.cell .code }
```python
transfer_model.eval()
with torch.no_grad():
    output = transfer_model(test_tensor.to(device))         # [1, 3]
    pred_label = output.argmax(dim=1).item()                # integer label

# Map class index to class name
pred_class = class_names[pred_label]  # e.g., "rock", "paper", "scissors"

# De-normalize and plot
mean = torch.tensor(imagenet_mean).view(3, 1, 1)
std = torch.tensor(imagenet_std).view(3, 1, 1)

img = test_tensor.squeeze(0).cpu() * std + mean
img_np = img.permute(1, 2, 0).clamp(0, 1).numpy()

plt.figure(figsize=(4, 4))
plt.imshow(img_np)
plt.title(f"Prediction: {pred_class}")
plt.axis('off')
plt.show()
```
:::

::: {.cell .markdown }

In practice, for most machine learning problems, you wouldn't design or
train a convolutional neural network from scratch - you would use an
existing model that suits your needs (does well on ImageNet, size is
right) and fine-tune it on your own data.

:::
