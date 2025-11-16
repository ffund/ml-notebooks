---
title: 'Neural networks for music classification'
author: 'Fraida Fund'
jupyter:
  anaconda-cloud: {}
  colab:
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  accelerator: 'GPU'
  gpuClass: 'standard'
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .markdown }
# Assignment: Neural Networks for Music Classification

_Fraida Fund_

:::

::: {.cell .markdown }
**TODO**: Edit this cell to fill in your NYU Net ID and your name:

-   **Net ID**:
-   **Name**:
:::

::: {.cell .markdown }

⚠️ **Note**: This experiment is designed to run on a Google Colab **GPU** runtime. You should use a GPU runtime on Colab to work on this assignment. You should not run it outside of Google Colab, and you should not run it in a CPU runtime. However, if you have been using Colab GPU runtimes a lot, you may be alerted that you have exhausted the "free" compute units allocated to you by Google Colab. We will have some limited availability of GPU time during the last week before the deadline, for students who have no compute units available.

:::

::: {.cell .markdown }
In this assignment, we will look at an audio classification problem.
Given a sample of music, we want to determine which instrument (e.g.
trumpet, violin, piano) is playing.

*This assignment is closely based on one by Sundeep Rangan, from his
[IntroML GitHub repo](https://github.com/sdrangan/introml/).*
:::

::: {.cell .code }
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
```
:::

::: {.cell .markdown }
## Audio Feature Extraction with Librosa

The key to audio classification is to extract the correct features. The
`librosa` package in python has a rich set of methods for extracting the
features of audio samples commonly used in machine learning tasks, such
as speech recognition and sound classification.
:::

::: {.cell .code }
```python
import librosa
import librosa.display
import librosa.feature
```
:::

::: {.cell .markdown }
In this lab, we will use a set of music samples from the website:

<http://theremin.music.uiowa.edu>

This website has a great set of samples for audio processing.

We will use the `wget` command to retrieve one file to our Google Colab
storage area. (We can run `wget` and many other basic Linux commands in
Colab by prefixing them with a `!` or `%`.)
:::

::: {.cell .code }
```python
!wget "http://theremin.music.uiowa.edu/sound files/MIS/Woodwinds/sopranosaxophone/SopSax.Vib.pp.C6Eb6.aiff"
```
:::

::: {.cell .markdown }
Now, if you click on the small folder icon on the far left of the Colab
interface, you can see the files in your Colab storage. You should see
the "SopSax.Vib.pp.C6Eb6.aiff" file appear there.
:::

::: {.cell .markdown }
In order to listen to this file, we'll first convert it into the `wav`
format. Again, we'll use a magic command to run a basic command-line
utility: `ffmpeg`, a powerful tool for working with audio and video
files.
:::

::: {.cell .code }
```python
aiff_file = 'SopSax.Vib.pp.C6Eb6.aiff'
wav_file = 'SopSax.Vib.pp.C6Eb6.wav'

!ffmpeg -y -i $aiff_file $wav_file
```
:::

::: {.cell .markdown }
Now, we can play the file directly from Colab. If you press the ▶️
button, you will hear a soprano saxaphone (with vibrato) playing four
notes (C, C\#, D, Eb).
:::

::: {.cell .code }
```python
import IPython.display as ipd
ipd.Audio(wav_file) 
```
:::

::: {.cell .markdown }
Next, use `librosa` command `librosa.load` to read the audio file with
filename `audio_file` and get the samples `y` and sample rate `sr`.
:::

::: {.cell .code }
```python
y, sr = librosa.load(aiff_file)
```
:::

::: {.cell .markdown }
Feature engineering from audio files is an entire subject in its own
right. A commonly used set of features are called the Mel Frequency
Cepstral Coefficients (MFCCs). These are derived from the so-called mel
spectrogram, which is something like a regular spectrogram, but the
power and frequency are represented in log scale, which more naturally
aligns with human perceptual processing.

You can run the code below to display the mel spectrogram from the audio
sample.

You can easily see the four notes played in the audio track. You also
see the \'harmonics\' of each notes, which are other tones at integer
multiples of the fundamental frequency of each note.
:::

::: {.cell .code }
```python
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
librosa.display.specshow(librosa.amplitude_to_db(S),
                         y_axis='mel', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()
```
:::

::: {.cell .markdown }
## Preparing the Data

Using the MFCC features described above, [Prof. Juan
Bello](http://steinhardt.nyu.edu/faculty/Juan_Pablo_Bello) at NYU
Steinhardt and his former PhD student Eric Humphrey have created a
complete data set that can used for instrument classification.
Essentially, they collected a number of data files from the website
above. For each audio file, the segmented the track into notes and then
extracted 120 MFCCs for each note. The goal is to recognize the
instrument from the 120 MFCCs. The process of feature extraction is
quite involved. So, we will just use their processed data.
:::

::: {.cell .markdown }
To retrieve their data, visit

[https://github.com/marl/dl4mir-tutorial/tree/master](https://github.com/marl/dl4mir-tutorial/tree/master)

and note the password listed on that page. Click on the link for
"Instrument Dataset", enter the password, click on
`instrument_dataset` to open the folder, and download it. 
(You can "direct download" straight from this site, you don't
need a Dropbox account.) Depending on your laptop OS and on how you download
the data, you may need to "unzip" or otherwise extract the four `.npy` files
from an archive.

Then, upload the files to your Google Colab storage: click on the folder
icon on the left to see your storage, if it isn't already open, and
then click on "Upload".

🛑 Wait until *all* uploads have completed and the orange "circles"
indicating uploads in progress are *gone*. (The training data especially
will take some time to upload.) 🛑
:::

::: {.cell .markdown }
Then, load the files with:
:::

::: {.cell .code }
```python
Xtr = np.load('uiowa_train_data.npy')
ytr = np.load('uiowa_train_labels.npy')
Xts = np.load('uiowa_test_data.npy')
yts = np.load('uiowa_test_labels.npy')
```
:::




::: {.cell .markdown }
Examine the data you have just loaded in:

-   How many training samples are there?
-   How many test samples are there?
-   What is the number of features for each sample?
-   How many classes (i.e. instruments) are there?

Write some code to find these values and print them.
:::

::: {.cell .code }
```python
#grade (write your code in this cell and DO NOT DELETE THIS LINE)

# TODO -  get basic details of the data
# compute these values from the data, don't hard-code them
n_tr    = ...
n_ts    = ...
n_feat  = ...
n_class = ...
```
:::


::: {.cell .code }
```python
# now print those details
print("Num training= %d" % n_tr)
print("Num test=     %d" % n_ts)
print("Num features= %d" % n_feat)
print("Num classes=  %d" % n_class)
```
:::

::: {.cell .markdown }

Convert the data to Pytorch "tensor" format:

:::


::: {.cell .code }
```python
Xtr_tensor = torch.tensor(Xtr, dtype=torch.float32)
Xts_tensor = torch.tensor(Xts, dtype=torch.float32)
ytr_tensor = torch.tensor(ytr, dtype=torch.long)
yts_tensor = torch.tensor(yts, dtype=torch.long)
```
:::


::: {.cell .markdown }

Then, standardize the training and test data, `Xtr_tensor` and `Xts_tensor`, by
removing the mean of each feature and scaling to unit variance. Save the standardized
data as `Xtr_scale` and `Xts_scale`, respectively.

Although you will scale both the training and test data, you should make sure that 
both are scaled according to the mean and variance statistics from the
*training data only*.

You can use `torch.mean()` and `torch.std()` to compute the statistics of the data - 
but make sure to specify the dimension, `dim`!

`<small>`{=html}Standardizing the input data can make the gradient
descent work better, by making the loss function "easier" to
descend.`</small>`{=html}

:::

::: {.cell .code }
```python
#grade (write your code in this cell and DO NOT DELETE THIS LINE)

# TODO - Standardize the training and test data
Xtr_scale = ...
Xts_scale = ...

```
:::

::: {.cell .markdown }

Use the standardized data to create a `TensorDataset`, then use that to create training and test `DataLoader`s. 
Use a batch size of 128. Shuffle the training data, but not the test data.

:::

::: {.cell .code }
```python
# TODO - Create the data loaders
batch_size = 128
train_ds = TensorDataset(Xtr_scale, ytr_tensor)
test_ds = TensorDataset(Xts_scale, yts_tensor)
# train_loader = DataLoader(...)
# test_loader = DataLoader(...)


```
:::


::: {.cell .markdown collapsed="true" }
## Building a Neural Network Classifier

Following the example in the demos you have seen, you will define 
a neural network in Pytorch with:

-   `nh=256` hidden units in a single dense hidden layer
-   `sigmoid` activation at hidden units
-   select the input and output shapes according
    to the problem requirements. Use the variables you defined earlier 
    (`n_tr`, `n_ts`, `n_feat`, `n_class`) as applicable, rather than 
    hard-coding numbers.
-   your model output should be a logit.


as a class named `InstrumentNet`.

Then create `model` as an instance of this class.

:::


::: {.cell .code }
```python
#grade (write your code in this cell and DO NOT DELETE THIS LINE)

# TODO - define the InstrumentNet
class InstrumentNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(InstrumentNet, self).__init__()
        # fill in details here!

    def forward(self, x):
        # fill in details here!

# create model as an instance of InstrumentNet
# model = ...
```
:::

::: {.cell .markdown}

Then, move `model` to the GPU device.

:::

::: {.cell .code}
```python
# TODO - create the model and move to device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# now move model to the device
```
:::


::: {.cell .markdown }

Create a `criterion` and an `optimizer` to train the model parameters. 

* Use an appropriate loss function for the `criterion`, considering that it is a 
multi-class classification problem and the model output is a logit.
* For the `optimizer`, use the Adam optimizer with a
learning rate of 0.001
:::

::: {.cell .code }
```python
# TODO - create criterion and optimizer
# criterion = ...
# optimizer = ...
```
:::

::: {.cell .markdown}

Next, fill in the implementation of 
a training function `train_model` that trains your model for one epoch! 

Make sure to move each batch of data to `device` as you get it from the data loader. The function should return the 
training accuracy and training loss as estimated during training.

:::

::: {.cell .code}
```python
#grade (write your code in this cell and DO NOT DELETE THIS LINE)

# TODO - Training function
def train_model(model, criterion, optimizer, device, train_loader):

    # fill in details here
    return train_acc, train_loss
```
:::


::: {.cell .markdown }

Also, define an `eval_model` function. This function should accept a data 
loader `eval_loader`, and should return the accuracy and loss on that data.

Make sure to move each batch of data to `device` as you get it from the data loader. The function should return the 
training accuracy and training loss as estimated during training.

:::

::: {.cell .code}
```python
#grade (write your code in this cell and DO NOT DELETE THIS LINE)

# TODO - Evaluation function
def eval_model(model, criterion, device, eval_loader):

    # fill in details here
    return eval_acc, eval_loss
```
:::

::: {.cell .markdown }

Now, you will use `train_model` to train your model for 10 epochs. 
At the end of each epoch, use `eval_model` to evaluate your model on the test data.
Your final accuracy should be greater than 99%.

Print the training and test accuracy at the end of each epoch.

Save the training and test accuracy and loss after each epoch, to 
fill in the `results` dictionary as indicated below. 

:::

::: {.cell .code }
```python

# TODO - fit model and save training history per epoch
n_epochs = 10
results = {
    'train_acc': [],
    'test_acc': [],
    'train_loss': [],
    'test_loss': [],
}

# ...
```
:::


::: {.cell .markdown }
Plot the training and test loss values on the same plot. 
You should see that the training loss is
steadily decreasing. Use the [`semilogy`
plot](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.semilogy.html)
so that the y-axis is log scale.

Make sure to label each axis, and each series (training vs. validation/test).
:::

::: {.cell .code }
```python
# TODO - plot the training and validation loss in one plot
```
:::

::: {.cell .markdown }
## Varying training hyperparameters

One challenge in training neural networks is the selection of the **training hyperparameters**, 
for example: 

* learning rate
* learning rate decay schedule
* batch size
* optimizer-specific hyperparameters (for example, the `Adam` optimizer we have been using has `beta_1`, `beta_2`, and `epsilon` hyperparameters)

and this challenge is further complicated by the fact that all of these training hyperparameters interact with one another.

(Note: **training hyperparameters** are distinct from **model hyperparameters**, like the number of hidden units or layers.)
:::

::: {.cell .markdown }

Sometimes, the choice of training hyperparameters affects whether or not the model will find an acceptable set of weights at all - i.e. whether the optimizer converges.

It's more often the case, though, that **for a given model**, we can arrive at a set of weights that have similar performance in many different ways, i.e. with different combinations of optimizer hyperparameters. 

However, the *training cost* in both **time** and **energy** will be very much affected.

In this section, we will explore these further.

:::


::: {.cell .markdown }


In the next cell, we will repeat the process above, but we will repeat this in a loop,
once for each of the learning rates shown in the vector `rates`. In each iteration of the loop:

-   create a new instance of the `model` (using the class you defined earlier), and move it to `device`. Create a new `optimizer`. Use the Adam optimizer with the learning rate specific to this iteration
-   call the training function `train_model` that you wrote earlier to train the model for 20 epochs (make sure you are training a *new* model in each iteration, and not *continuing* the training of a model created already outside the loop). At the end of each epoch, use the `eval_model` function that you wrote earlier to evaluate the model on the test set.
-   save the history of training and test accuracy and loss

:::

::: {.cell .code }
```python
# TODO - iterate over learning rates
rates = [0.1, 0.01,0.001,0.0001]
n_epochs = 20

results = {}

for lr in rates:
    results[lr] = {
        'train_acc': [],
        'test_acc': [],
        'train_loss': [],
        'test_loss': []
    }
    # fill in the rest of the implementation

```
:::

::: {.cell .markdown }

Plot the *training* loss vs. the epoch number for all of the learning
rates on one graph (use `semilogy` again), using a different color for each learning rate. 
Do *not* include the test loss in this plot.

You should see that the lower learning rates are more stable, but converge slower, while with a
learning rate that is too high, the gradient descent may fail to move
towards weights that decrease the loss function.

Make sure to label each axis, and each series.

**Comment on the results.** Given that all other optimizer hyperparameters are fixed, what is the effect of varying learning rate on the training process?

:::

::: {.cell .code }
```python
# TODO - plot showing the training process for different learning rates
```
:::


::: {.cell .markdown }

In the previous example, we trained each model for a fixed number of epochs. Now, we'll explore what happens when we vary the training hyperparameters, but train each model to the same validation **accuracy target**. We will consider:

* how much *time* it takes to achieve that accuracy target ("time to accuracy")
* how much *energy* it takes to achieve that accuracy target ("energy to accuracy")
* and the *test accuracy* for the model, given that it is trained to the specified validation accuracy target

:::

::: {.cell .markdown }

#### Energy consumption


To do this, first we will need some way to measure the energy used to train the model. We will use [Zeus](https://ml.energy/zeus/overview/), a Python package developed by researchers at the University of Michigan, to measure the GPU energy  consumption.


First, install the package:

:::

::: {.cell .code}
```python
!pip install zeus
```
:::


::: {.cell .markdown }

Then, import it, and start an instance of a monitor, specifying the GPU that it should monitor:

:::

::: {.cell .code}
```python
from zeus.monitor import ZeusMonitor

monitor = ZeusMonitor(gpu_indices=[0])
```
:::

::: {.cell .markdown }

When you want to measure GPU energy usage, you will:

* start a "monitoring window"
* do your GPU-intensive computation (e.g. call `train_model`)
* stop the "monitoring window"

and then you can get the time and total energy used by the GPU in the monitoring window.

:::


::: {.cell .markdown}

Try it now - call `train_model` to train whatever `model` is currently in scope from previous cells for one more epoch.
Observe the measured time and energy.

:::

::: {.cell .code}
```python
monitor.begin_window("test")

# call your train function here for one epoch

measurement = monitor.end_window("test")
print("Measured time (s)  :" , measurement.time)
print("Measured energy (J):" , measurement.total_energy)
```
:::



::: {.cell .markdown }

#### Train to Specified Accuracy

Next, we need a way to train a model until we achieve our desired validation accuracy. 

:::

::: {.cell .markdown}

First, we need a validation set! If we "train until desired validation accuracy", we are using the validation set to make decisions about model training. Therefore, we can't use the test set as a validation set here. We need to split a separate validation set out of the test set.

Use `random_split` to split out 20% of the training data for a validation set (as shown in e.g. the Colab lesson 
on convolutional neural networks!).

Then, create a new `train_loader` and a `val_loader` data loader. 
Use a batch size of 128 and shuffle the training data.

:::


::: {.cell .code}
```python
# TODO - split out validation subset
# train_ds_split, val_ds_split = random_split(...)
# val_loader = ...
# train_loader = ...

```
:::

::: {.cell .markdown}

Now, we can train to a specified accuracy!

* In the following cell, we define a `train_to_accuracy` function, which will accept the following arguments: a `max_epochs` value, a `threshold`, and a `patience` value. (In addition to passing: `model`, `criterion`, `optimizer`, `device`, `train_loader`, `val_loader`, `test_loader`, `train_model`, and `eval_model`.)
* Inside the function, you will call `train_model` to train a model for up to `max_epochs`. At the end of each epoch, you will use `eval_model` to evaluate the model on the *validation* data (not the *test* data!)
* If the model's validation accuracy is higher than the `threshold` for `patience` epochs in a row, stop training  even if the `max_epochs` is not reached. 
* At the end of training, use `eval_model` to evaluate the model on the test data.
* The default values of `max_epochs`, `threshold`, and `patience` are given below, but other values may be passed as arguments at runtime.

Fill in the implementation below. You only need to return the final test data statistics - you don't need to save these per epoch. You will also return the number of training epochs required to reach the desired accuracy.

:::

::: {.cell .code}
```python
#grade (write your code in this cell and DO NOT DELETE THIS LINE)

# TODO - define the train_to_accuracy function

def train_to_accuracy(
    max_epochs=100, threshold=0.9, patience=3,
    model=None, criterion=None, optimizer=None, device=None,
    train_loader=None, val_loader=None, test_loader=None,
    train_model=None, eval_model=None
):
    # fill in details here
    return test_acc, test_loss, epochs

```
:::


::: {.cell .markdown}
Try it! Test your implementation. As some basic "sanity checks", make sure that:

* The smallest number of epochs it may use is `patience`, and the largest number is `max_epochs`
* The final training accuracy is at least `threshold`

but also, check your implementation thoroughly and make sure it behaves as expected.

:::


::: {.cell .code}
```python
# train_to_accuracy(max_epochs=20, threshold=0.95, patience=5,
#    model=model, criterion=criterion, optimizer=optimizer, device=device,
#    train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
#    train_model=train_model, eval_model=eval_model)
```
:::




::: {.cell .markdown}
### See how TTA/ETA varies with learning rate, batch size

:::


::: {.cell .markdown }


Now, you will repeat your model preparation and fitting code - with your new `train_to_accuracy` function - but in a loop. First, you will iterate over different learning rates.

In each iteration, you will prepare a model (with the appropriate training hyperparameters) and train it until:

* either it has achieved **0.98 accuracy for 3 epoches in a row** on the 20% validation subset of the training data,
* or, it has trained for 100 epochs

whichever comes FIRST. 

For each iteration, you will record:

* the training hyperparameters (learning rate, batch size)
* the number of epochs of training needed to achieve the target validation accuracy
* the accuracy on the *test* data (not the validation data!)
* the GPU energy and time to train the model to the desired validation accuracy, as computed by a `zeus` measurement window that starts just before the call to `train_to_accuracy` and ends just afterward.

:::

::: {.cell .code }
```python
# TODO - iterate over learning rates and get TTA/ETA

# default learning rate and batch size -
lr = 0.001
batch_size = 128

metrics_vs_lr = []
for lr in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]


    # TODO - set up model
    # set up optimizer with specified learning rate

    # start measurement 
    # call train_to_accuracy
    # end measurement
    
    # save results in a dictionary
    # model_metrics = {
    #    'batch_size': ...,
    #    'learning_rate': ...,
    #    'epochs': ...,
    #    'test_accuracy': ...,
    #    'test_loss': ...,
    #    'total_energy': ..., 
    #    'train_time': ...
    # }

    # TODO - append model_metrics dictionary to the metrics_vs_lr list
```
:::


::: {.cell .markdown}

Look at the output and make sure it is reasonable:

:::

::: {.cell .code}
```python
print(metrics_vs_lr)
```
:::


::: {.cell .markdown}

Next, you will visualize the results. 

Create a figure with four subplots. In each subplot, create a bar plot with learning rate on the horizontal axis and (1) Time to accuracy, (2) Energy to accuracy, (3) Test accuracy, (4) Epochs, on the vertical axis on each subplot, respectively. Use an appropriate vertical range for each subplot. Label all axes.



:::


::: {.cell .code}
```python
# TODO - visualize effect of varying learning rate, when training to a target accuracy

```
:::


::: {.cell .markdown}

**Comment on the results**: Given that the model is trained to a target validation accuracy, what is the effect of the learning rate on the training process *in this example*?

:::



::: {.cell .markdown }


Now, you will repeat, with a loop over different batch sizes! This time, you will
need to create a new `train_loader` inside each iteration. (You don't have to change
the data loader for the evaluation or test sets, since they won't affect the 
training process.)

:::

::: {.cell .code }
```python
# TODO - iterate over batch size and get TTA/ETA

# default learning rate and batch size -
lr = 0.001
batch_size = 128

metrics_vs_bs = []
for batch_size in [32,  128, 512, 2048]:

    # TODO - set up model
    # set up optimizer with specified learning rate
    # set up a new train_loader with specified batch size 
    # (use the "split" of training data that does not include the validation set!)

    # start measurement 
    # call train_to_accuracy
    # end measurement
    
    # save results in a dictionary
    # model_metrics = {
    #    'batch_size': ...,
    #    'learning_rate': ...,
    #    'epochs': ...,
    #    'test_accuracy': ...,
    #    'test_loss': ...,
    #    'total_energy': ..., 
    #    'train_time': ...
    # }

    # TODO - append model_metrics dictionary to the metrics_vs_bs list
```
:::


::: {.cell .markdown}

Next, you will visualize the results. 

Create a figure with four subplots. In each subplot, create a bar plot with batch size on the horizontal axis and (1) Time to accuracy, (2) Energy to accuracy, (3) Test accuracy, (4) Epochs, on the vertical axis on each subplot, respectively. Use an appropriate vertical range for each subplot. Label all axes.



:::

::: {.cell .markdown}

Look at the output and make sure it is reasonable:

:::

::: {.cell .code}
```python
print(metrics_vs_bs)
```
:::


::: {.cell .code}
```python
# TODO - visualize effect of varying batch size, when training to a target accuracy

```
:::


::: {.cell .markdown}

**Comment on the results**: Given that the model is trained to a target validation accuracy, what is the effect of the batch size on the training process *in this example*? What do you observe about how time and energy *per epoch* and number of epochs required varies with batch size? 


:::
