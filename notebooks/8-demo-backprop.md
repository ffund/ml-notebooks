---
title: 'Demo: Backpropagation'
author: 'Fraida Fund'
jupyter:
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
    version: 3.7.3
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .markdown }
## Demo: Backpropagation

*Fraida Fund*
:::

::: {.cell .markdown }
In this demo, we will show how to use backpropagation to train a simple
neural network for regression.
:::

::: {.cell .code }
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
```
:::

::: {.cell .markdown }
### Generate data
:::

::: {.cell .markdown }
For this demo, we will use synthetic data which relates an individual's
income to the number of years of education they have, and their
seniority in their position.

The data is from "An Introduction to Statistical Learning, with
applications in R" (Springer, 2013) by G. James, D. Witten, T. Hastie
and R. Tibshirani.
:::

::: {.cell .code }
```python
df = pd.read_csv('https://www.statlearning.com/s/Income2.csv', index_col=0)
df.head()
```
:::

::: {.cell .code }
```python
X = np.vstack((df['Education'], df['Seniority'])).T
y = df['Income']
y = np.resize(y, (30,1))
```
:::

::: {.cell .code }
```python
print(X.shape)
print(y.shape)
```
:::

::: {.cell .code }
```python
sns.scatterplot(x=X[:,0], y=X[:,1], hue=y.ravel())
plt.xlabel("Education");
plt.ylabel("Seniority");
```
:::

::: {.cell .code }
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
```
:::

::: {.cell .code }
```python
# make numbers small so they are easier to work with
x1_max = np.amax(abs(X_train[:,0]))
x2_max = np.amax(abs(X_train[:,1]))
y_max = np.amax(abs(y_train))

X_train[:,0] = X_train[:,0]/x1_max
X_train[:,1] = X_train[:,1]/x2_max
y_train = y_train/y_max

X_test[:,0] = X_test[:,0]/x1_max
X_test[:,1] = X_test[:,1]/x2_max
y_test = (y_test)/y_max
```
:::

::: {.cell .markdown }
### Construct a network

Now, we will construct a neural network for this dataset.
:::

::: {.cell .markdown }
#### Structure of the network

First, we will describe the structure of the network, and draw it. We
have a *regression* problem with

-   Number of inputs: $N_I = 2$
-   Number of outputs: $N_O = 1$
-   Number of training samples $N=15$

and we will use a single hidden layer with three hidden units, $N_H=3$.
:::

::: {.cell .code }
```python
inputLayerSize  = X.shape[1]
outputLayerSize = y.shape[1]
hiddenLayerSize = 3
```
:::

::: {.cell .markdown }
#### Drawing the network

We can use the `networkx` library in Python to draw the network.
:::

::: {.cell .code }
```python
import networkx as nx

nodePos = {}
G=nx.Graph()
graphHeight = max(inputLayerSize, outputLayerSize, hiddenLayerSize)

# create nodes and note their positions
for n in range(inputLayerSize):
  nodePos['x'+str(n+1)]=(1, n)
  G.add_node('x'+str(n+1))
for n in range(outputLayerSize):
  nodePos['o'+str(n+1)]=(5, n)
  G.add_node('o'+str(n+1))
for n in range(hiddenLayerSize):
  nodePos['h'+str(n+1)]=(3, n)
  G.add_node('h'+str(n+1))

# add edges
for n in range(hiddenLayerSize):
  for m in range(inputLayerSize):
    G.add_edge('x' + str(m+1), 'h' + str(n+1))
  for m in range(outputLayerSize):
    G.add_edge('h' + str(n+1), 'o' + str(m+1))
```
:::

::: {.cell .code }
```python
nx.draw_networkx(G, pos=nodePos, 
                 node_size=1000, node_color='pink')
plt.margins(0.2, 0.2)
```
:::

::: {.cell .markdown }
#### Including a bias term

We will also add a *bias node* at each layer. This simplifies the
computation of weights by adding an extra input whose value is always 1.
The bias term then comes from the weight applied to that input.
:::

::: {.cell .code }
```python
nodePos['xb']=(1, inputLayerSize)
G.add_node('xb')
for n in range(hiddenLayerSize):
  G.add_edge('xb', 'h' + str(n+1))

nodePos['hb']=(3, hiddenLayerSize)
G.add_node('hb')
for n in range(outputLayerSize):
  G.add_edge('hb', 'o' + str(n+1))

nx.draw_networkx(G, pos=nodePos, 
                 node_size=1000, node_color='pink')
plt.margins(0.2, 0.2)
```
:::

::: {.cell .markdown }
We will augment the data to add the bias "feature" $x_b$, whose value
is always a 1:
:::

::: {.cell .code }
```python
X_train_aug = np.column_stack((np.ones((X_train.shape[0],1)), X_train))
X_train_aug
```
:::

::: {.cell .code }
```python
X_test_aug = np.hstack((np.ones((X_test.shape[0],1)), X_test))
X_test_aug
```
:::

::: {.cell .markdown }
#### Defining inputs and outputs in the network

Next, we will define the input and output at each node in the network.
:::

::: {.cell .markdown }
At every node in the hidden layer, the input to the activation function
is the weighted sum of inputs to the node:

$$z_j = \sum_i w_{j,i} x_i$$

including the intput from the bias node $x_b$, which is always 1.

The output of a hidden node is the output of an activation function
applied to that weighted sum of inputs. We will use a sigmoid activation
function:

$$u_H = g_{H}(z_H) = \sigma(z_H) = \frac{1}{1 + e^{-z_H}}$$
:::

::: {.cell .code }
```python
def sigmoid(z):
    return 1/(1+np.exp(-z))
```
:::

::: {.cell .markdown }
In the output layer, we use the weighted sum of inputs to the node from
the hidden layer (i.e. weighted sum of each $u_i$ from hidden node $i$):

$$z_j = \sum_i w_{j,i} u_i$$

including one $u_{hb}$ from the hidden layer bias node, which always
outputs a 1.

Then, since this is a regression problem, at the output layer we will
use the identity function on $z_O$ to get the final output $u_O$:

$$\hat{y} = u_O = g_{O}(z_O)=z_O$$
:::

::: {.cell .markdown }
#### Summary of network variables and dimensions

The table below summarizes the variables in the network, and their
dimensions.
:::

::: {.cell .markdown }
   Code Symbol   Math Symbol               Definition                    Dimensions
  ------------- ------------- ------------------------------------- --------------------
       `X`          $$X$$      Training data, each row is a sample    ($N$, $N_I$ + 1)
       `y`          $$y$$            Labels of training data            ($N$, $N_O$)
      `W_H`       $$W_{H}$$           Hidden layer weights           ($N_I + 1$, $N_H$)
      `W_O`       $$W_{O}$$           Output layer weights           ($N_H + 1$, $N_O$)
      `z_H`       $$z_{H}$$     Input to hidden layer activation        ($N$, $N_H$)
      `u_H`       $$u_{H}$$     Output of hidden layer activation       ($N$, $N_H$)
      `z_O`       $$z_{O}$$     Input to output layer activation        ($N$, $N_O$)
      `u_O`       $$u_{O}$$     Output of output layer activation       ($N$, $N_O$)

where for this problem $N=15$, $N_I = 2$, $N_O = 1$, $N_H = 3$.
:::

::: {.cell .code }
```python
nx.draw_networkx(G, pos=nodePos, 
                 node_size=1000, node_color='pink')
plt.margins(0.2, 0.2)
```
:::

::: {.cell .markdown }
#### Initialize weights

To start, we will initialize all weights to random values:
:::

::: {.cell .code }
```python
np.random.seed(9)
W_H = np.random.randn(inputLayerSize+1, hiddenLayerSize)
W_O = np.random.randn(hiddenLayerSize+1, outputLayerSize)
# the extra 1 is because of the bias nodes
```
:::

::: {.cell .code }
```python
W_H
```
:::

::: {.cell .code }
```python
W_O
```
:::

::: {.cell .markdown }
### Feed values forward

To start training our neural network, we will feed the training values
through the network and observe the output. This is known as the
"forward pass".
:::

::: {.cell .markdown }
#### Computations in the forward pass

In the forward pass, we proceed through the computation graph from the
input to the output. At each stage, we will compute a function of the
output of the previous stage.

First, we will compute the linear transform on the inputs, using the
current weights at the hidden layer:

$$z_H =  W_H^T X $$

Then, we will use $z_H$ and apply the activation function at the hidden
layer:

$$u_H = g_{H}(z_H)$$

The next step is to compute the linear transform at the output layer,
using $u_H$ as input:

$$z_O =  W_O^T [1, u_H] $$

Finally, we use $z_O$ and compute the response function at the output:

$$u_O = g_{O}(z_O)  $$

We will save $z_H$, $u_H$, $z_O$, and $u_O$ in each iteration, since
these values will be used to compute the gradient in the backward pass.
:::

::: {.cell .code }
```python
# feed inputs though network
def forward(X, W_H, W_O):
    # linear transform at input to hidden layer
    z_H = np.dot(X, W_H) 
    # activation function at hidden layer
    u_H = sigmoid(z_H)
    # add an extra "ones" column for the bias term at the output layer
    u_H_b = np.column_stack((np.ones((u_H.shape[0],1)), u_H))
    # linear transform at input to output layer
    z_O = np.dot(u_H_b, W_O)
    # output function
    u_O = z_O
    return z_H, u_H, z_O, u_O
```
:::

::: {.cell .markdown }
#### Execution of first forward pass

We are ready to execute the first forward pass.

With our initial random weights, the output of the network on the
training set is:
:::

::: {.cell .code }
```python
z_H, u_H, z_O, u_O = forward(X_train_aug, W_H, W_O)
y_hat = u_O
y_hat
```
:::

::: {.cell .markdown }
#### Computing loss

With our first set of predictions in hand, we can compute the loss
function for this network. Since this is a regression problem, we will
use the mean squared error loss function. For $N$ input samples,

$$L(W) = \frac{1}{2}\sum_N (y_i - u_{Oi})^2$$

where $W = (W_H, W_O)$.

(The $\frac{1}{2}$ factor in front is for convenience, so that when we
compute the derivative in order to do gradient descent, we won't have
to carry a $2$ factor with us throughout our computations.)
:::

::: {.cell .code }
```python
def loss_function(y_true, y_hat):
    # compute loss function for a given prediction
    return 0.5*np.sum((y_true-y_hat)**2, axis=0)
```
:::

::: {.cell .markdown }
Then the training loss of our network for this training set, using the
initial random weights, is:
:::

::: {.cell .code }
```python
l=loss_function(y_train, y_hat)
print(l)
```
:::

::: {.cell .markdown }
### Propagate error backwards

To update the weights in our neural network, we will use gradient
descent to move in the direction of steepest (infinitesimal) decrease in
our loss function. We will therefore need to compute the gradient of the
loss function with respect to each weight:
$\frac{\partial L(W)}{\partial W}$.
:::

::: {.cell .markdown }
Backpropagation is an iterative procedure for efficient computation of
gradients using the chain rule. In backpropagation, we compute gradients
starting from the output of the network, and work our way back towards
the inputs.
:::

::: {.cell .markdown }
#### Digression: chain rule on a computational graph
:::

::: {.cell .markdown }
Suppose we have a computational graph, representing the composite
function $f(g(h(x)))$.

To compute the output, we do a *forward pass* on the computational
graph:
:::

::: {.cell .markdown }
![image.png](data:image/png;base64,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)
:::

::: {.cell .markdown }
To compute the derivative via chain rule, we do a *backward pass* on the
computational graph, where we get the derivative of each node with
respect to its inputs.
:::

::: {.cell .markdown }
![image.png](data:image/png;base64,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)
:::

::: {.cell .markdown }
We will use a similar method to train our neural network.
:::

::: {.cell .markdown }

We will execute the backpropagation algorithm on our neural network as
follows:

**Step 1**:  Apply input (or batch of inputs) to network and propagate values
    forward. (Sum is over all inputs to node $j$)

$$z_j = \sum_i w_{j,i} u_i, \quad u_j = g(z_j)$$

**Step 2**:   Evaluate $\delta_k = \frac{\delta L}{\delta z_k}$ for all output
    units.

For a regression network with a linear activation function at the output
nodes, $\delta_k = \frac{\partial L}{\partial z_k} = -(y_n - z_{k})$

**Step 3**:   For each hidden unit, we we “backpropagate” the $\delta$s from all
    outputs of a hidden unit to get $\delta_j$ for that hidden unit.
    (Sum is over all outputs of node $j$)

$$
\begin{aligned}
\delta_j & = \frac{\delta L}{\delta z_j} \\
 & = \sum_k \frac{\delta L}{\delta z_k}\frac{\delta z_k}{\delta z_j} \\
 & = \sum_k  \delta_k \frac{\delta z_k}{\delta z_j} \\
 & = \sum_k  \delta_k w_{k,j} g'(z_j) \\
 & = g'(z_j) \sum_k w_{k,j} \delta_k
\end{aligned}
$$

**Step 4**:   Use $\frac{\partial L_n}{\partial w_{j,i}} = \delta_j u_i$ to
    evaluate derivatives with respect to weights at all nodes.

:::

::: {.cell .markdown }
![image.png](data:image/png;base64,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)
:::

::: {.cell .markdown }
We are using a sigmoid activation function at the hidden units, so we
will write a function `sigmoidPrime` to compute the derivative of a
sigmoid with respect to its input.
:::

::: {.cell .code }
```python
def sigmoidPrime(z):
    # derivative of sigmoid function
    return np.exp(-z)/((1+np.exp(-z))**2)
```
:::

::: {.cell .markdown }
Finally, we can write our `backprop` function:
:::

::: {.cell .code }
```python
def backprop(X, y, W_H, W_O, z_H, u_H, z_O, u_O):
    # compute backpropagation error at output layer  
    deltaO = -(y-z_O)

    # compute backpropagation error at hidden layer  
    # (note that we don't propagate back error of bias node)
    deltaH = np.dot(deltaO, W_O[1:hiddenLayerSize+1].T)*sigmoidPrime(z_H)

    # compute derivative w.r.t. weights at output layer
    u_H_b = np.hstack((np.ones((u_H.shape[0],1)), u_H))
    dLdwO = np.dot(u_H_b.T, deltaO).reshape(hiddenLayerSize+1,outputLayerSize)

    # compute derivative w.r.t. weights at hidden layer
    dLdwH = np.dot(X.T, deltaH)
    # return derivatives
    return dLdwO, dLdwH
```
:::

::: {.cell .code }
```python
dLdwO, dLdwH = backprop(X_train_aug, y_train, W_H, W_O, z_H, u_H, z_O, u_O)
```
:::

::: {.cell .code }
```python
print(dLdwO)
print(dLdwH)
```
:::

::: {.cell .code }
```python
print(W_H)
print(W_O)
```
:::

::: {.cell .markdown }
### Apply gradient descent
:::

::: {.cell .markdown }
Now that we have computed the gradient of the loss function with respect
to each weight, we can use gradient descent. The gradient points towards
the direction of greatest (infinitesimal) increase, so we will move each
weight in the opposite direction.
:::

::: {.cell .code }
```python
learningRate = 0.01
W_H = W_H - learningRate*dLdwH
W_O = W_O - learningRate*dLdwO
```
:::

::: {.cell .markdown }
After we complete another forward pass and compute loss for the new
weights, we can see that the loss has decreased.
:::

::: {.cell .code }
```python
z_H, u_H, z_O, u_O = forward(X_train_aug, W_H, W_O)
l_new = loss_function(y_train, u_O) 
print(l, l_new)
```
:::

::: {.cell .markdown }
### Iterate
:::

::: {.cell .markdown }
To train our network, we will repeat this procedure for some number of
*epochs*.
:::

::: {.cell .code }
```python
epochs = 1000
y_hat = np.zeros((epochs, y_train.shape[0], y_train.shape[1]))
L_train = np.zeros(epochs)
L_test = np.zeros(epochs)
```
:::

::: {.cell .code }
```python
learningRate = 0.05
np.random.seed(321)

W_H_init = np.random.randn(inputLayerSize+1, hiddenLayerSize)
W_O_init = np.random.randn(hiddenLayerSize+1, outputLayerSize)

W_H = W_H_init
W_O = W_O_init

for e in range(epochs):

  # get loss on test set
  # this is not used to update weights, 
  # we only do this for visualization
  z_H, u_H, z_O, u_O = forward(X_test_aug, W_H, W_O)
  L_test[e] = loss_function(y_test, u_O) 

  # forward pass for this epoch
  z_H, u_H, z_O, u_O = forward(X_train_aug, W_H, W_O)
  # get loss on training set
  L_train[e] = loss_function(y_train, u_O) 
  y_hat[e] = u_O

  # backward pass for this epoch
  dLdwO, dLdwH = backprop(X_train_aug, y_train, W_H, W_O, z_H, u_H, z_O, u_O)
  # gradient descent
  W_H = W_H - learningRate*dLdwH
  W_O = W_O - learningRate*dLdwO
```
:::

::: {.cell .markdown }
Finally, we will visualize the performance of the network as it trains.
:::

::: {.cell .code }
```python
# plot output values vs iterations
colors = sns.color_palette("hls", y_train.shape[0])
plt.figure(figsize=(15,6))
plt.subplot(121)
plt.ylim(np.amin(y_train*y_max)-10, np.amax(y_train*y_max)+10)
for n in range(y_train.shape[0]):
  plt.axhline(y=y_max*y_train[n], linestyle='--', color=colors[n])
  sns.lineplot(x=np.arange(0, epochs), y=y_max*y_hat[:,n].ravel(), color=colors[n])
plt.xlabel("Epoch");
plt.ylabel("Predicted income for training samples, $\hat{y}$");

plt.subplot(122)
p = sns.lineplot(x=np.arange(0, epochs), y=L_train, color='blue', label='Training')
p = sns.lineplot(x=np.arange(0, epochs), y=L_test, color='red', label='Test')
p.set(yscale="log")
plt.xlabel("Epoch");
plt.ylabel("Loss");
```
:::

::: {.cell .markdown }
### Analyze results
:::

::: {.cell .markdown }
Here is a plot of true vs. predicted values, for the training and test
data:
:::

::: {.cell .code }
```python
z_H, u_H, z_O, u_O = forward(X_train_aug, W_H, W_O)
y_pred_train = u_O[:,0]
z_H, u_H, z_O, u_O = forward(X_test_aug, W_H, W_O)
y_pred_test = u_O[:,0]
```
:::

::: {.cell .code }
```python
sns.scatterplot(x=y_train[:,0], y = y_pred_train, label='Training data');
sns.scatterplot(x=y_test[:,0], y = y_pred_test, label='Test data');
plt.xlim(0,1);
plt.ylim(0,1);
plt.xlabel('True value');
plt.ylabel('Predicted value');
```
:::

::: {.cell .markdown }
(a perfect regression result would be the diagonal line $y=x$.)
:::
