import tensorflow as tf

from .unscented import Unscented


class NNR(Unscented):
    """
    Neural Network Regressor
    This class defines a fully connected neural network transformation
    for the GDSM model.
    It uses the Unscented class as basis for propagation and loss estimation.

    The form of the function is
    f(x) = h(x) + sigma_noise * epsilon
    epsilon being a normally distributed variable and h a neural network.
    """

    def __init__(self, input_dim, output_dim, hidden_layers=1, hidden_dim=None,
                 activation_function=tf.tanh,
                 sigma_init=1., sigma_noise=0., trainable_noise=False,
                 loss_samples=100, trainable=True,
                 dtype='float64', name='NNR', **kwargs):
        """
        Initialises the neural network transformation object.

        Args:
            input_dim : Dimension of the input
            output_dim : Dimension of the output
            hidden_layers : Number of hidden layers of the network (default 1)
            hidden_dim : Dimension of the hidden layers, if it is an integer it applies to
                all the hidden layers, if it is a list it defines the size
                for each hidden layers. (default max(input_dim, output_dim))
            activation_function : Activation function to use for hidden layers
                (default tanh)
            sigma_init : Initial scale, at initialisation, of the transformation.
                Usually it can be set to 1 for the transition function,
                for the observation it should be the standard deviation of your
                observations to simplify training. (default 1.)
            sigma_noise : Initial scale of the Gaussian noise (default 0.)
            trainable_noise : Boolean indicating if the noise scale is trainable.
                (default True)
            trainable : Boolean indicating if the complete transformation is trainable.
                (default True)
            loss_samples : Number of samples to use when estimating the loss function.
                (default 100)
            dtype : TensorFlow type used for the function's variables
                and operations (default 'float64')
            name : Name of the function (default 'NNR')
            **kwargs : Supplementary arguments for the parent class.
        """

        self.input_dim = input_dim
        self.output_dim = output_dim
        self._hidden_layers = hidden_layers
        self._activation_function = activation_function
        self._dtype = dtype

        hidden_dim = hidden_dim if hidden_dim is not None else \
            max(input_dim, output_dim)
        if not hasattr(hidden_dim, '__len__'):
            hidden_dim = self._hidden_layers * [hidden_dim]
        self._hidden_dim = hidden_dim

        with tf.name_scope(name):
            # Create function variables
            prev_dim = self.input_dim
            self._weights = []
            for i in range(self._hidden_layers + 1):
                shape = [prev_dim,
                         self.output_dim if i == self._hidden_layers else
                         self._hidden_dim[i]]

                biases = tf.zeros([1, shape[1]], dtype=self._dtype)
                stddev = (2. / sum(shape)) ** 0.5 \
                    * 1. if i < self._hidden_layers else sigma_init
                weights = tf.random_normal(
                    shape, stddev=stddev, dtype=self._dtype)
                initial_value = tf.concat([biases, weights], 0)
                self._weights.append(tf.Variable(
                    initial_value, trainable, name='weights'))

                prev_dim = shape[1]

            super().__init__(sigma_noise=sigma_noise,
                             trainable_noise=trainable_noise and trainable,
                             loss_samples=loss_samples, **kwargs)

    def variables_to_save(self):
        """
        Implementation of the function variables_to_save
        required by GDSM. (see template.py)
        """
        return self._weights + super().variables_to_save()

    def apply_to(self, inputs):
        """
        Implementation of the function apply_to
        required by the Unscented class.
        """

        batch_shape = tf.shape(inputs)[:-2]

        inputs = tf.reshape(inputs, [-1, self.input_dim])
        ones = tf.ones([tf.shape(inputs)[0], 1], dtype=self._dtype)

        for i, weights in enumerate(self._weights):

            # Add constant and multiply with weights
            inputs = tf.matmul(tf.concat([ones, inputs], 1), weights)

            # Activation layer of not output
            if i < self._hidden_layers:
                inputs = self._activation_function(inputs)

        outputs_shape = tf.concat([batch_shape, [self.output_dim, 1]], 0)
        return tf.reshape(inputs, outputs_shape)
