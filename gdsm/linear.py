import tensorflow as tf


class Linear:
    """
    This class defines a linear transformation for the GDSM model.

    The form of the function is
    f(x) = mu + W.x + sigma_noise * epsilon
    epsilon being a normally distributed variable.
    """

    def __init__(self, input_dim, output_dim,
                 sigma_init=1., sigma_noise=0.,
                 trainable_noise=True, trainable=True,
                 dtype='float64', name='Linear'):
        """
        Initialises the linear transformation object.

        Args:
            input_dim : Dimension of the input
            output_dim : Dimension of the output
            sigma_init : Initial scale, at initialisation, of the transformation.
                The matrix `W` is just randomly initialised, this
                parameters enforces that the sum of the values on
                each line is equal to it.
                Usually it can be set to 1 for the transition function,
                for the observation it should be the standard deviation of your
                observations to simplify training. (default 1.)
            sigma_noise : Initial scale of the Gaussian noise (default 0.)
            trainable_noise : Boolean indicating if the noise scale is trainable.
                (default True)
            trainable : Boolean indicating if the complete transformation is trainable.
                (default True)
            dtype : TensorFlow type used for the function's variables
                and operations (default 'float64')
            name : Name of the function (default 'Linear')
        """

        self.input_dim = input_dim
        self.output_dim = output_dim
        self._dtype = dtype

        with tf.name_scope(name):
            self._sigma_noise = tf.Variable(
                sigma_noise * tf.ones([self.output_dim], dtype=self._dtype),
                trainable and trainable_noise, name='sigma_noise')
            self._sigma_noise2 = tf.square(self._sigma_noise)
            self._added_noise_var = tf.diag(self._sigma_noise2)

            temp = tf.random_normal([self.input_dim, self.output_dim],
                                    dtype=self._dtype)
            temp *= sigma_init / tf.reduce_sum(temp, axis=0, keep_dims=True)
            self._W = tf.Variable(tf.transpose(temp), trainable, name='W')
            self._mu = tf.Variable(
                tf.zeros([self.output_dim, 1], dtype=self._dtype),
                trainable, name='mu')

    def variables_to_save(self):
        """
        Implementation of the function variables_to_save
        required by GDSM. (see template.py)
        """
        return [self._sigma_noise,
                self._W,
                self._mu]

    def propagate(self, mu, Sigma):
        """
        Implementation of the function propagate
        required by GDSM. (see template.py)
        """

        do_batch = mu.get_shape().ndims != 2
        batch_shape = tf.shape(mu)[:-2]

        # Compute output mean
        if do_batch:
            temp = tf.reshape(mu, [-1, self.input_dim])
            temp = tf.matmul(temp, self._W, transpose_b=True)
            temp = tf.reshape(
                temp, tf.concat([batch_shape, [self.output_dim, 1]], 0))
            m = temp + self._mu
        else:
            m = tf.matmul(self._W, mu) + self._mu

        # Compute output variance
        if do_batch:
            temp = tf.reshape(Sigma, [-1, self.input_dim])
            temp = tf.matmul(temp, self._W, transpose_b=True)
            temp = tf.reshape(
                temp,
                tf.concat([batch_shape, [self.input_dim, self.output_dim]], 0))
            C_oi = tf.matrix_transpose(temp)
            temp = tf.reshape(C_oi, [-1, self.input_dim])
            temp = tf.matmul(temp, self._W, transpose_b=True)
            temp = tf.reshape(
                temp, tf.concat(
                    [batch_shape, [self.output_dim, self.output_dim]], 0))
            C = temp + self._added_noise_var
        else:
            C_oi = tf.matmul(self._W, Sigma)
            C = tf.matmul(C_oi, self._W, transpose_b=True) \
                + self._added_noise_var

        return m, C, C_oi

    def loss_function(self, m_i, C_i, m_o, C_o=None, C_io=None):
        """
        Implementation of the function loss_function
        required by GDSM. (see template.py)
        """

        do_batch = m_i.get_shape().ndims != 2
        batch_shape = tf.shape(m_i)[:-2]

        if do_batch:
            temp = tf.reshape(m_i, [-1, self.input_dim])
            temp = tf.matmul(temp, self._W, transpose_b=True)
            temp = tf.reshape(
                temp, tf.concat([batch_shape, [self.output_dim, 1]], 0))
            out = temp + self._mu
        else:
            out = tf.matmul(self._W, m_i) + self._mu

        # Compute output variance
        if do_batch:
            temp = tf.reshape(C_i, [-1, self.input_dim])
            temp = tf.matmul(temp, self._W, transpose_b=True)
            temp = tf.reshape(
                temp, tf.concat(
                    [batch_shape, [self.input_dim, self.output_dim]], 0))
            temp = tf.matrix_transpose(temp)
            temp = tf.reshape(temp, [-1, self.input_dim])
            temp = tf.matmul(temp, self._W, transpose_b=True)
            temp = tf.reshape(
                temp, tf.concat(
                    [batch_shape, [self.output_dim, self.output_dim]], 0))
        else:
            temp = tf.matmul(tf.matmul(self._W, C_i),
                             self._W, transpose_b=True)

        cost = tf.matrix_diag_part(temp) + tf.squeeze(tf.square(out), -1) \
            + tf.squeeze(tf.square(m_o), -1) - 2 * tf.squeeze(m_o * out, -1)

        if C_o is not None:
            if do_batch:
                temp = tf.reshape(
                    tf.matrix_transpose(C_io), [-1, self.input_dim])
                temp = tf.matmul(temp, self._W, transpose_b=True)
                temp = tf.reshape(
                    temp, tf.concat(
                        [batch_shape, [self.output_dim, self.output_dim]], 0))
            else:
                temp = tf.matmul(self._W, C_io)

            cost += tf.matrix_diag_part(C_o) - 2 * tf.matrix_diag_part(temp)

        return cost / self._sigma_noise2 \
            + 2 * tf.log(tf.abs(self._sigma_noise))

    def sample(self, inputs):
        """
        Implementation of the function sample
        required by GDSM. (see template.py)
        """

        batch_shape = tf.shape(inputs)[:-2]
        outputs_shape = tf.concat([batch_shape, [self.output_dim, 1]], 0)

        temp = tf.reshape(inputs, [-1, self.input_dim])
        temp = tf.matmul(temp, self._W, transpose_b=True)
        temp = tf.reshape(temp, outputs_shape)

        m = temp + self._mu

        return m + tf.random_normal(outputs_shape, dtype=self._dtype) \
            * tf.expand_dims(self._sigma_noise, -1)
