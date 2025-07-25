---
title: 'Learning the Slash dataset'
author: 'Fraida Fund'
jupyter:
  colab:
    toc_visible: true
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .markdown }
# Demo: Convolutional neural networks on the "slash" dataset

*Fraida Fund*
:::

::: {.cell .markdown }
In this demo, we'll look at an example of a task that is difficult for
"classical" machine learning models, and difficult for fully connected
neural networks, but easy for convolutional neural networks.
:::

::: {.cell .code }
```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from sklearn.model_selection import train_test_split
from sklearn import ensemble, neighbors, linear_model, svm


from ipywidgets import interactive, Layout
import ipywidgets as widgets

```
:::


::: {.cell .markdown }
## The slash dataset

The "slash" dataset, developed by [Sophie
Searcy](https://soph.info/slash-data), is a set of images, each of which
includes a "slash" on a background of random noise. The data is
divided into two classes according to whether the slash is downward
facing or upward facing.
:::

::: {.cell .code }
```python
def gen_example(size=20, label=0):

    max_s_pattern = int(size // 4)
    s_pattern = 4
    pattern = 1- np.eye(s_pattern)
    if label:
        pattern = pattern[:, ::-1]
    ex = np.ones((size,size))
    point_loc = np.random.randint(0, size - s_pattern + 1,
                                  size=(2, ))  # random x,y point
    ex[point_loc[0]:point_loc[0] + s_pattern, point_loc[1]:point_loc[1] +
       s_pattern] = pattern  # set point to
    ex = ex + .5*(np.random.rand(size, size) - .5)
    np.clip(ex,0.,1., out=ex)
    return ex
```
:::

::: {.cell .code }
```python
examples = []

n_side = 30
n_ex = 500 #number of examples in each class

for i in range(n_ex):
    examples.append(gen_example(size=n_side, label=0))
    examples.append(gen_example(size=n_side, label=1))
    
y = np.array([0,1]*n_ex)
x = np.stack(examples)
```
:::

::: {.cell .code }
```python
plt.figure(figsize=(18,4))

n_print = 10 # number of examples to show

ex_indices = np.random.choice(len(y), n_print, replace=False)
for i, index in enumerate(ex_indices):
    plt.subplot(1, n_print, i+1, )
    plt.imshow(x[index,...], cmap='gray')
    plt.title(f"y = {y[index]}")
```
:::

::: {.cell .markdown }
We'l prepare training and test data in two formats:

-   "flat" for traditional ML models and fully connected neural
    networks, which don't care about the spatial arrangement of the
    features.
-   "image" for convolutional neural networks, which expect the input to have a depth, height, and width. (The "depth" dimension is 1 for these grayscale images.)
:::

::: {.cell .code }
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.25)

x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

x_train_img = np.expand_dims(x_train, axis=1)  # shape [B, 1, H, W]
x_test_img = np.expand_dims(x_test, axis=1)    # shape [B, 1, H, W]
```
:::

::: {.cell .code }
```python
print("Flat data shape:  ", x_train_flat.shape)
print("Image data shape: ", x_train_img.shape)
```
:::

::: {.cell .markdown }
The feature data is in the range 0 to 1:
:::

::: {.cell .code }
```python
x.min(), x.max()
```
:::

::: {.cell .markdown }
## Train logistic regression, random forest, KNN, SVM models
:::

::: {.cell .markdown }
Next, we'll try to train some classic ML models on this dataset.
:::

::: {.cell .code }
```python
models = {
    "Logistic\n Regression": linear_model.LogisticRegression(),
    "KNN-1": neighbors.KNeighborsClassifier(n_neighbors=1),
    "KNN-3": neighbors.KNeighborsClassifier(n_neighbors=3),
    "Random\n Forest": ensemble.RandomForestClassifier(n_estimators=100),
    "SVM -\n Linear": svm.SVC(kernel="linear"),
    "SVM -\n RBF kernel": svm.SVC(kernel="rbf")
}
```
:::

::: {.cell .code }
```python
results = []

for model_name in models.keys():    
    model = models[model_name]
    model.fit(x_train_flat, y_train)
    
    train_score = model.score(x_train_flat, y_train)
    test_score = model.score(x_test_flat, y_test)   
    
    results.append({"model": model_name, "train_score": train_score, "test_score": test_score})
```
:::

::: {.cell .code }
```python
results_df = pd.DataFrame(results)

plt.figure(figsize =(10,10));

plt.subplot(2,1,1)
sns.barplot(x=results_df.sort_values('test_score')['model'], y=results_df.sort_values('test_score')['train_score']);
plt.ylim(0,1);
plt.xlabel("")

plt.subplot(2,1,2)
sns.barplot(x=results_df.sort_values('test_score')['model'], y=results_df.sort_values('test_score')['test_score']);
plt.ylim(0,1);
```
:::

::: {.cell .markdown }
Are these the results we expected? Why or why not?
:::

::: {.cell .markdown }
Do *any* of these models do a good job of learning whether a slash is
forward-facing or backward-facing?
:::

::: {.cell .markdown }
## Train a fully connected neural network
:::


::: {.cell .markdown }

Next, we will set up a fully connected neural network in Pytorch:

* it will accept a flat array of all pixels in the slash image as input (we'll pass that when we create an instance of the model)
* it will have three hidden layers, each with 64 fully connected units
* it will use a ReLU activation at the hidden layers
* it will have one output unit with a sigmoid activation

:::


::: {.cell .code }
```python
class FCNet(nn.Module):
    def __init__(self, nin, nh1=64, nh2=64, nh3=64, nout=1):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(nin, nh1)
        self.fc2 = nn.Linear(nh1, nh2)
        self.fc3 = nn.Linear(nh2, nh3)
        self.out = nn.Linear(nh3, nout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.out(x))
        return x

```
:::

::: {.cell .markdown }

We'll set up the `DataLoader` to feed images in batches for training and evaluation:

:::


::: {.cell .code }
```python
train_dataset = TensorDataset(
    torch.tensor(x_train_flat, dtype=torch.float32), 
    torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


test_dataset = TensorDataset(
    torch.tensor(x_test_flat, dtype=torch.float32), 
    torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```
:::

::: {.cell .markdown }

We're going to prefer to train this model on a GPU. So before we start, we will 

* check if a GPU is available (a "cuda" device) with `torch.cuda.is_available()`
* if a GPU is available, set `device` to the GPU device; otherwise set it to the CPU device
* create an instance of the fully connected model, and *move it to the device*

:::

::: {.cell .code }
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_fc = FCNet(nin=x_train_flat.shape[1]).to(device)
```
:::

::: {.cell .markdown }

Finally, we can specify the loss function and the optimizer.

:::

::: {.cell .code }
```python
criterion = nn.BCELoss()
optimizer = optim.Adam(model_fc.parameters(), lr=0.001)

```
:::


::: {.cell .markdown }

Here's our model:
:::

::: {.cell .code }
```python
model_fc
```
:::

::: {.cell .markdown }

We will train it for 100 epochs. 

Within each epoch, we will iterate over the training `DataLoader`, passing a batch of data at a time to the model.  We will *move the data to the `device`* - since the model is moved to GPU (if one is avaliable), the data also must be moved to GPU.

Then, we do our forward pass and backwards pass, and update the model parameters.

:::

::: {.cell .code }
```python
n_epochs = 100
model_fc.train()
for epoch in range(n_epochs):
    epoch_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model_fc(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
```
:::

::: {.cell .markdown }

Once our model is trained, let's evaluate it on both the training and test data:

:::


::: {.cell .code }
```python
model_fc.eval()
train_preds, train_targets = [], []
test_preds, test_targets = [], []

with torch.no_grad():
    for xb, yb in train_loader:
        xb = xb.to(device)
        preds = model_fc(xb)
        train_preds.append(preds.cpu())
        train_targets.append(yb)

    for xb, yb in test_loader:
        xb = xb.to(device)
        preds = model_fc(xb)
        test_preds.append(preds.cpu())
        test_targets.append(yb)

train_preds = torch.cat(train_preds).numpy()
train_targets = torch.cat(train_targets).numpy()
test_preds = torch.cat(test_preds).numpy()
test_targets = torch.cat(test_targets).numpy()

train_score = ((train_preds > 0.5).astype(int).flatten() == train_targets.flatten()).mean()
test_score = ((test_preds > 0.5).astype(int).flatten() == test_targets.flatten()).mean()

print(f"Train Accuracy: {train_score:.4f}")
print(f"Test Accuracy: {test_score:.4f}")
```
:::

::: {.cell .code }
```python
results.append({"model": 'FC Neural Net', "train_score": train_score, "test_score": test_score})
```
:::

::: {.cell .code }
```python
results_df = pd.DataFrame(results)

plt.figure(figsize =(11,10));

plt.subplot(2,1,1)
sns.barplot(x=results_df.sort_values('test_score')['model'], y=results_df.sort_values('test_score')['train_score']);
plt.ylim(0,1);
plt.xlabel("")

plt.subplot(2,1,2)
sns.barplot(x=results_df.sort_values('test_score')['model'], y=results_df.sort_values('test_score')['test_score']);
plt.ylim(0,1);
```
:::

::: {.cell .markdown }

Although this model has tens of thousands of parameters, it does not do very well on this data.

:::


::: {.cell .code }
```python
total_params = sum(
	param.numel() for param in model_fc.parameters()
)
trainable_params = sum(
	p.numel() for p in model_fc.parameters() if p.requires_grad
)

print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")
```
:::

::: {.cell .markdown }
## Train a convolutional neural network
:::

::: {.cell .markdown }

Now, we'll try a convolutional neural network:

* it will accept as input a 3D volume, although in this case the depth dimension is 1 (there are no color channels)
* it will have the following sequence of hidden layers:
    * a convolution layer with 8 filters, 3x3 size, zero-padding 1, and ReLU activation
    * a max pooling layer with 2x2 filter size
    * a batch normalization layer
    * another convolution layer with the same settings
    * a "global" average pooling layer, that returns the average of the entire output from each filter in the previous layer
    * and finally, a fully connected layer connected to the output unit
* it will have one output unit with a sigmoid activation

:::


::: {.cell .code }
```python
class ConvNet(nn.Module):
    def __init__(self, in_channels=1, filters=8):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, filters, kernel_size=3, padding=1, bias=False)  # [B, in_channels, H, W] -> [B, filters, H, W]
        self.pool = nn.MaxPool2d(kernel_size=2)  # [B, filters, H, W] -> [B, filters, H/2, W/2]
        self.bn = nn.BatchNorm2d(filters)  # [B, filters, H/2, W/2] -> [B, filters, H/2, W/2]
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)  # [B, filters, H/2, W/2] -> [B, filters, H/2, W/2]
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # [B, filters, H/2, W/2] -> [B, filters, 1, 1]
        self.fc = nn.Linear(filters, 1)  # [B, filters] -> [B, 1]

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.bn(x)
        x = F.relu(self.conv2(x))
        x = self.global_avg_pool(x)  # shape: [B, filters, 1, 1]
        x = x.view(x.size(0), -1)    # flatten to [B, filters]
        x = torch.sigmoid(self.fc(x))
        return x

```
:::


::: {.cell .markdown }

We'll set up the `DataLoader` to feed images in batches for training and evaluation - this time, with spatial dimension

:::


::: {.cell .code }
```python
train_dataset = TensorDataset(
    torch.tensor(x_train_img, dtype=torch.float32), 
    torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


test_dataset = TensorDataset(
    torch.tensor(x_test_img, dtype=torch.float32), 
    torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```
:::


::: {.cell .markdown }

As before, we'll put the model on GPU if available:

:::

::: {.cell .code }
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_conv = ConvNet(in_channels=1).to(device)
```
:::

::: {.cell .markdown }

Finally, we can specify the loss function and the optimizer.

:::

::: {.cell .code }
```python
criterion = nn.BCELoss()
optimizer = optim.Adam(model_conv.parameters(), lr=0.001)

```
:::


::: {.cell .markdown }

Here's our model:
:::

::: {.cell .code }
```python
model_conv
```
:::

::: {.cell .markdown }

and train the model:

:::

::: {.cell .code }
```python
n_epochs = 100
model_conv.train()
for epoch in range(n_epochs):
    epoch_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model_conv(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
```
:::

::: {.cell .markdown }

Once our model is trained, let's evaluate it on both the training and test data:

:::


::: {.cell .code }
```python
model_conv.eval()
train_preds, train_targets = [], []
test_preds, test_targets = [], []

with torch.no_grad():
    for xb, yb in train_loader:
        xb = xb.to(device)
        preds = model_conv(xb)
        train_preds.append(preds.cpu())
        train_targets.append(yb)

    for xb, yb in test_loader:
        xb = xb.to(device)
        preds = model_conv(xb)
        test_preds.append(preds.cpu())
        test_targets.append(yb)

train_preds = torch.cat(train_preds).numpy()
train_targets = torch.cat(train_targets).numpy()
test_preds = torch.cat(test_preds).numpy()
test_targets = torch.cat(test_targets).numpy()

train_score = ((train_preds > 0.5).astype(int).flatten() == train_targets.flatten()).mean()
test_score = ((test_preds > 0.5).astype(int).flatten() == test_targets.flatten()).mean()

print(f"Train Accuracy: {train_score:.4f}")
print(f"Test Accuracy: {test_score:.4f}")
```
:::


::: {.cell .code }
```python
results.append({"model": 'ConvNet', "train_score": train_score, "test_score": test_score})
```
:::

::: {.cell .code }
```python
results_df = pd.DataFrame(results)

plt.figure(figsize =(12,10));

plt.subplot(2,1,1)
sns.barplot(x=results_df.sort_values('test_score')['model'], y=results_df.sort_values('test_score')['train_score']);
plt.ylim(0,1);
plt.xlabel("")

plt.subplot(2,1,2)
sns.barplot(x=results_df.sort_values('test_score')['model'], y=results_df.sort_values('test_score')['test_score']);
plt.ylim(0,1);
```
:::

::: {.cell .markdown }
## Using the same model on different slashes
:::

::: {.cell .markdown }
Not only did our convolutional network learn forward and backward
slashes - it can even generalize to slightly different forward and
backward slashes.

Let's generate data with heavier background noise, and longer slashes:
:::

::: {.cell .code }
```python
noise_scale = 0.65
s_pattern = 15
def gen_example_different(size=20, label=0):

    max_s_pattern = int(size // 4)
    pattern = 1- np.eye(s_pattern)
    if label:
        pattern = pattern[:, ::-1]
    ex = np.ones((size,size))
    point_loc = np.random.randint(0, size - s_pattern + 1,
                                  size=(2, ))  # random x,y point
    ex[point_loc[0]:point_loc[0] + s_pattern, point_loc[1]:point_loc[1] +
       s_pattern] = pattern  # set point to
    ex = ex + noise_scale*(np.random.rand(size, size) - .5)
    np.clip(ex,0.,1., out=ex)
    return ex

examples = []

n_side = 30
n_ex = 50 #number of examples in each class

for i in range(n_ex):
    examples.append(gen_example_different(size=n_side, label=0))
    examples.append(gen_example_different(size=n_side, label=1))
    
y_new = np.array([0,1]*n_ex)
x_new = np.stack(examples)

plt.figure(figsize=(18,4))

n_print = 10 # number of examples to show

ex_indices = np.random.choice(len(y_new), n_print, replace=False)
for i, index in enumerate(ex_indices):
    plt.subplot(1, n_print, i+1, )
    plt.imshow(x_new[index,...], cmap='gray')
    plt.title(f"y = {y_new[index]}")
```
:::

::: {.cell .code }
```python
plt.figure(figsize=(18,4))

for i, index in enumerate(ex_indices):
    plt.subplot(1, n_print, i+1, )
    plt.imshow(x_new[index,...], cmap='gray')

    x_new_tensor = torch.tensor(x_new[index][np.newaxis, np.newaxis, :, :], dtype=torch.float32).to(device)
    with torch.no_grad():
        yhat = model_conv(x_new_tensor).cpu().item()  # scalar output

    plt.title(f"yhat = {yhat:.2f}")
```
:::

::: {.cell .code }
```python
x_new_tensor = torch.tensor(x_new[:, np.newaxis, :, :], dtype=torch.float32).to(device)  # shape [N, 1, H, W]
y_new_tensor = torch.tensor(y_new, dtype=torch.float32).unsqueeze(1).to(device)          # shape [N, 1]

model_conv.eval()
with torch.no_grad():
    y_pred = model_conv(x_new_tensor)
    y_pred_class = (y_pred > 0.5).float()
    correct = (y_pred_class == y_new_tensor).float().sum().item()
    total = y_new_tensor.size(0)
    new_test_score = correct / total

print(f"New test accuracy: {new_test_score:.4f}")
```
:::

::: {.cell .markdown }
What about forward and backward slashes at different angles?
:::

::: {.cell .code }
```python
max_rot = 10
def gen_example_rotated(size=20, label=0):

    max_s_pattern = int(size // 4)
    s_pattern = 15
    pattern = 1- np.eye(s_pattern)
    if label:
        pattern = pattern[:, ::-1]
    ex = np.ones((size,size))
    point_loc = np.random.randint(0, size - s_pattern + 1,   size=(2, ))  
    ex[point_loc[0]:point_loc[0] + s_pattern, point_loc[1]:point_loc[1] + s_pattern] = pattern  
    rot_angle = np.random.uniform(-max_rot, max_rot)
    ex = scipy.ndimage.rotate(ex, angle=rot_angle, cval=1, reshape = False)
    ex = ex + noise_scale*(np.random.rand(size, size) - .5)

    np.clip(ex,0.,1., out=ex)
    return ex

examples = []

n_side = 30
n_ex = 50 #number of examples in each class

for i in range(n_ex):
    examples.append(gen_example_rotated(size=n_side, label=0))
    examples.append(gen_example_rotated(size=n_side, label=1))
    
y_new = np.array([0,1]*n_ex)
x_new = np.stack(examples)

plt.figure(figsize=(18,4))

n_print = 10 # number of examples to show

ex_indices = np.random.choice(len(y_new), n_print, replace=False)
for i, index in enumerate(ex_indices):
    plt.subplot(1, n_print, i+1, )
    plt.imshow(x_new[index,...], cmap='gray')
    plt.title(f"y = {y_new[index]}")
```
:::

::: {.cell .code }
```python
plt.figure(figsize=(18,4))

for i, index in enumerate(ex_indices):
    plt.subplot(1, n_print, i+1, )
    plt.imshow(x_new[index,...], cmap='gray')

    x_new_tensor = torch.tensor(x_new[index][np.newaxis, np.newaxis, :, :], dtype=torch.float32).to(device)
    with torch.no_grad():
        yhat = model_conv(x_new_tensor).cpu().item()  # scalar output

    plt.title(f"yhat = {yhat:.2f}")
```
:::

::: {.cell .code }
```python
x_new_tensor = torch.tensor(x_new[:, np.newaxis, :, :], dtype=torch.float32).to(device)  # shape [N, 1, H, W]
y_new_tensor = torch.tensor(y_new, dtype=torch.float32).unsqueeze(1).to(device)          # shape [N, 1]

model_conv.eval()
with torch.no_grad():
    y_pred = model_conv(x_new_tensor)
    y_pred_class = (y_pred > 0.5).float()
    correct = (y_pred_class == y_new_tensor).float().sum().item()
    total = y_new_tensor.size(0)
    new_test_score = correct / total

print(f"New test accuracy: {new_test_score:.4f}")
```
:::
