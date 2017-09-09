# TensorFlow Gaussian Dynamical System Modelling

GSDM allows you to do parameter estimation on time series using all kind of dynamical models.

You just need to specify the transition and the observation function (follow examples from `gdsm/linear.py` or `gdsm/template.py`).
The E-step is done using variants of the Kalman filter (Unscented, Monte-Carlo...), when defining a dynamics function you can define how to propagate a Gaussian through it.
The module also comes with pre-coded dynamics functions such as a linear transformation, a fully connected neural network, etc...

Here is an example of usage...
``` python
import tensorflow as tf
import matplotlib.pyplot as plt
from gdsm import GDSM, GP
from my_dataset import get_sequences

# Load data
seq_train, seq_test = get_sequences()

# Create computational graph (using default tf graph)
f = GP(input_dim=2, output_dim=2, dataset_size=16)
g = GP(input_dim=2, output_dim=1, dataset_size=16)
model = GDSM(f, g)

# Replace default optimisation op
opt = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
m_step = opt.minimize(model.loss)
model.set_m_step(m_step)

with tf.Session() as sess:
    # Start training
    loss_history = \
        model.train_on_batch(seq_train, em_steps=10, m_steps=20)
    
    # Plot evolution of the loss during training
    plt.plot(loss_history)
    plt.show()

    # From there you can use the trained model
    # to perform all sorts of operations
    # ...
```
