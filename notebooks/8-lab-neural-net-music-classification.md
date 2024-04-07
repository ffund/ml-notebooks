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

⚠️ **Note**: This experiment is designed to run on a Google Colab **GPU** runtime. You should use a GPU runtime on Colab to work on this assignment. You should not run it outside of Google Colab. However, if you have been using Colab GPU runtimes a lot, you may be alerted that you have exhausted the "free" compute units allocated to you by Google Colab. We will have some limited availability of GPU time during the last week before the deadline, for students who have no compute units available.

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
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import time
%matplotlib inline
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
## Downloading the Data

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

::: {.cell .code }
```python
# shuffle the training set 
# (when loaded in, samples are ordered by class)
p = np.random.permutation(Xtr.shape[0])
Xtr = Xtr[p,:]
ytr = ytr[p]
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
Then, standardize the training and test data, `Xtr` and `Xts`, by
removing the mean of each feature and scaling to unit variance.

You can do this manually, or using `sklearn`\'s
[StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).
(For an example showing how to use a `StandardScaler`, you can refer to
the notebook on regularization.)

Although you will scale both the training and test data, you should make sure that 
both are scaled according to the mean and variance statistics from the
*training data only*.

`<small>`{=html}Standardizing the input data can make the gradient
descent work better, by making the loss function "easier" to
descend.`</small>`{=html}
:::

::: {.cell .code }
```python
# TODO - Standardize the training and test data
Xtr_scale = ...
Xts_scale = ...

```
:::

::: {.cell .markdown collapsed="true" }
## Building a Neural Network Classifier

Following the example in the demos you have seen, clear the keras
session. Then, create a neural network `model` with:

-   `nh=256` hidden units in a single dense hidden layer
-   `sigmoid` activation at hidden units
-   select the input and output shapes, and output activation, according
    to the problem requirements. Use the variables you defined earlier 
    (`n_tr`, `n_ts`, `n_feat`, `n_class`) as applicable, rather than 
    hard-coding numbers.


Print the model summary.

:::

::: {.cell .code }
```python
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
import tensorflow.keras.backend as K
```
:::




::: {.cell .code }
```python
# TODO - construct the model
nh = 256
# model =  ...
# model.add( ...
 
```
:::

::: {.cell .code}
```python
# show the model summary
model.summary()
```
:::


::: {.cell .code}
```python
# you can also visualize the model with 
tf.keras.utils.plot_model(model, show_shapes=True)
```
:::


::: {.cell .markdown }
Create an optimizer and compile the model. Select the appropriate loss
function for this multi-class classification problem, and use an
accuracy metric. For the optimizer, use the Adam optimizer with a
learning rate of 0.001
:::

::: {.cell .code }
```python
# TODO - create optimizer and compile the model
# opt = ...
# model.compile(...)
```
:::

::: {.cell .markdown }
Fit the model for 10 epochs using the scaled data for both training
and validation, and save the training history in `hist.

Use the `validation_data` option to pass the *test*
data. (This is OK because we are not going to use this data
as part of the training process, such as for early stopping - 
we're just going to compute the accuracy on the data so that 
we can see how training and test loss changes as the model is 
trained.)


Use a batch size of 128. Your final accuracy should be greater than 99%.
:::

::: {.cell .code }
```python
# TODO - fit model and save training history
# hist = 
```
:::

::: {.cell .markdown }
Plot the training and validation accuracy saved in `hist.history`
dictionary, on the same plot. This gives one accuracy value per epoch.
You should see that the validation accuracy saturates around 99%. After
that it may "bounce around" a little due to the noise in the
stochastic mini-batch gradient descent.

Make sure to label each axis, and each series (training vs. validation/test).
:::

::: {.cell .code }
```python
# TODO - plot the training and validation accuracy in one plot
```
:::

::: {.cell .markdown }
Plot the training and validation loss values saved in the `hist.history`
dictionary, on the same plot. You should see that the training loss is
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


Repeat your model preparation and fitting code, but try four learning rates as shown in the vector `rates`. In each iteration of the loop:

-   use `K.clear_session()` to free up memory from models that are no longer in scope. (Note that this does not affect models that are still "in scope"!)
-   construct the network
-   select the optimizer. Use the Adam optimizer with the learning rate specific to this iteration
-   train the model for 20 epochs (make sure you are training a *new* model in each iteration, and not *continuing* the training of a model created already outside the loop)
-   save the history of training and validation accuracy and loss for
    this model

:::

::: {.cell .code }
```python
rates = [0.1, 0.01,0.001,0.0001]

# TODO - iterate over learning rates
```
:::

::: {.cell .markdown }

Plot the training loss vs. the epoch number for all of the learning
rates on one graph (use `semilogy` again). You should see that the lower
learning rates are more stable, but converge slower, while with a
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

**Note**: if you are running this experiment in a CPU-only runtime, you should skip this section on energy comsumption. Continue with the " `TrainToAccuracy` callback" section.


First, install the package:

:::

::: {.cell .code}
```python
!pip install zeus-ml
```
:::


::: {.cell .markdown }

Then, import it, and tell it to monitor your GPU:

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
* do your GPU-intensive computation (e.g. call `model.fit`)
* stop the "monitoring window"

and then you can get the time and total energy used by the GPU in the monitoring window.

:::


::: {.cell .markdown}
Try it now - this will just continue fitting whatever `model` is currently in scope from previous cells:
:::

::: {.cell .code}
```python
monitor.begin_window("test")
model.fit(Xtr_scale, ytr, epochs=5)
measurement = monitor.end_window("test")
print("Measured time (s)  :" , measurement.time)
print("Measured energy (J):" , measurement.total_energy)
```
:::



::: {.cell .markdown }

#### `TrainToAccuracy` callback

Next, we need a way to train a model until we achieve our desired validation accuracy. We will [write a callback function](https://www.tensorflow.org/guide/keras/writing_your_own_callbacks)
following these specifications:

* It will be called `TrainToAccuracy` and will accept two arguments: a `threshold` and a `patience` value.
* If the model's validation accuracy is higher than the `threshold` for `patience` epochs in a row, stop training. 
* In the `on_epoch_end` function, which will be called at the end of every epoch during training, you should get the current validation accuracy using `currect_acc = logs.get("val_accuracy")`. Then, set `self.model.stop_training = True` if the condition above is met.
* The default values of `threshold` and `patience` are given below, but other values may be passed as arguments at runtime.

Then, when you call `model.fit()`, you will add the `TrainToAccuracy` callback as in

```
callbacks=[TrainToAccuracy(threshold=0.98, patience=3)]
```

:::


::: {.cell .code}
```python
# TODO - write a callback function
class TrainToAccuracy(callbacks.Callback):

    def __init__(self, threshold=0.9, patience=5):
        self.threshold = threshold  # the desired accuracy threshold
        self.patience = patience # how many epochs to wait once hitting the threshold

    def on_epoch_end(self, epoch, logs=None):
        current_acc = logs.get("val_accuracy")
        # if conditions are met..
        # self.model.stop_training = True

```
:::


::: {.cell .markdown}
Try it! run the following cell to test your `TrainToAccuracy` callback. (This will just continue fitting whatever `model` is currently in scope.)
:::


::: {.cell .code}
```python
model.fit(Xtr_scale, ytr, epochs=100, validation_split = 0.2, callbacks=[TrainToAccuracy(threshold=0.95, patience=3)])
```
:::


::: {.cell .markdown}

Your model shouldn't *really* train for 100 epochs - it should stop training as soon as 95% validation accuracy is achieved for 3 epochs in a row! (Your "test" is not graded, you may change the `threshold` and `patience` values in this "test" call to `model.fit` in order to check your work.)

Note that since we are now using the validation set performance to *decide* when to stop training the model, we are no longer "allowed" to pass the test set as `validation_data`. The test set must never be used to make decisions during the model training process - only for evaluation of the final model. Instead, we specify that 20% of the training data should be held out as a validation set, and that is the validation accuracy that is used to determine when to stop training.

:::



::: {.cell .markdown}
### See how TTA/ETA varies with learning rate, batch size

:::


::: {.cell .markdown }


Now, you will repeat your model preparation and fitting code - with your new `TrainToAccuracy` callback - but in a loop. First, you will iterate over different learning rates.

In each iteration of each loop, you will prepare a model (with the appropriate training hyperparameters) and train it until:

* either it has achieved **0.98 accuracy for 3 epoches in a row** on a 20% validation subset of the training data,
* or, it has trained for 500 epochs

whichever comes FIRST. 

For each model, you will record:

* the training hyperparameters (learning rate, batch size)
* the number of epochs of training needed to achieve the target validation accuracy
* the accuracy on the *test* data (not the validation data!). After fitting the model, use `model.evaluate` and pass the scaled *test* data to get the test loss and test accuracy
* the GPU energy and time to train the model to the desired validation accuracy, as computed by a `zeus-ml` measurement window that starts just before `model.fit` and ends just after `model.fit`.

:::

::: {.cell .code }
```python

# TODO - iterate over learning rates and get TTA/ETA

# default learning rate and batch size -
lr = 0.001
batch_size = 128

metrics_vs_lr = []
for lr in [0.0001, 0.001, 0.01, 0.1]:

    # TODO - set up model, including appropriate optimizer hyperparameters

    # start measurement 
    # if on GPU runtime
    try: 
        monitor.begin_window("model_train")
    # if on GPU runtime, but last measurement window is still running
    except ValueError: 
        _ = monitor.end_window("model_train")
        monitor.begin_window("model_train")
    # if on CPU runtime
    except NameError: 
        print("Uh oh! You are not connected to a GPU runtime.") 
        start_time = time.time()


    # TODO - fit model on (scaled) training data 
    # until specified validation accuracy is achieved (don't use test data!)
    # but stop after 500 epochs even if validation accuracy is not achieved

    # end measurement 
    # if on GPU runtime
    try: 
        measurement = monitor.end_window("model_train")
    # if on CPU runtime
    except NameError:  
        total_time = time.time() - start_time

    # TODO - evaluate model on (scaled) test data
    
    # save results in a dictionary
    # model_metrics = {
    #    'batch_size': ...,
    #    'learning_rate': ...,
    #    'epochs': ...,
    #    'test_accuracy': ...,
    #    'total_energy': ..., # if on GPU runtime
    #    'train_time': ...
    # }

    # TODO - append model_metrics dictionary to the metrics_vs_lr list
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


Now, you will repeat, with a loop over different batch sizes - 

:::

::: {.cell .code }
```python

# TODO - iterate over batch size and get TTA/ETA

# default learning rate and batch size -
lr = 0.001
batch_size = 128

metrics_vs_bs = []
for batch_size in [64, 128, 256, 512, 1024, 2048, 4096, 8192]:

    # TODO - set up model, including appropriate optimizer hyperparameters

    # start measurement 
    # if on GPU runtime
    try: 
        monitor.begin_window("model_train")
    # if on GPU runtime, but last measurement window is still running
    except ValueError: 
        _ = monitor.end_window("model_train")
        monitor.begin_window("model_train")
    except NameError: 
        print("Uh oh! You are not connected to a GPU runtime.") 
        start_time = time.time()


    # TODO - fit model on (scaled) training data 
    # until specified validation accuracy is achieved (don't use test data!)
    # but stop after 500 epochs even if validation accuracy is not achieved

    # end measurement 
    # if on GPU runtime
    try: 
        measurement = monitor.end_window("model_train")
    except NameError:  
        total_time = time.time() - start_time

    # TODO - evaluate model on (scaled) test data
    
    # save results in a dictionary
    # model_metrics = {
    #    'batch_size': ...,
    #    'learning_rate': ...,
    #    'epochs': ...,
    #    'test_accuracy': ...,
    #    'total_energy': ..., # if on GPU runtime
    #    'train_time': ...
    # }

    # TODO - append model_metrics dictionary to the metrics_vs_bs list
```
:::


::: {.cell .markdown}

Next, you will visualize the results. 

Create a figure with four subplots. In each subplot, create a bar plot with batch size on the horizontal axis and (1) Time to accuracy, (2) Energy to accuracy, (3) Test accuracy, (4) Epochs, on the vertical axis on each subplot, respectively. Use an appropriate vertical range for each subplot. Label all axes.



:::


::: {.cell .code}
```python
# TODO - visualize effect of varying batch size, when training to a target accuracy

```
:::


::: {.cell .markdown}

**Comment on the results**: Given that the model is trained to a target validation accuracy, what is the effect of the batch size on the training process *in this example*? What do you observe about how time and energy *per epoch* and number of epochs required varies with batch size? 


:::
