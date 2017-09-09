import tensorflow as tf
import numpy as np


class Unscented:
    """
    Use this as a parent class in order to define a
    nonlinear transformation for the GDSM models.
    The function has the form
    f(x) = h(x) + `sigma_noise` * epsilon
    where epsilon is a normally distributed variable.

    This class uses the unscented transform (see Julier and Uhlmann
    "A new extension of the Kalman filter to nonlinear systems")
    to compute the Gaussian propagation required by GDSM.
    It uses a naive sampling approach to approximate the loss function.

    You need to define the function `apply_to` in order to define
    the dynamics (the function h in the above equation).
    `apply_to` must accept inputs of shape [... , `input_dim`, 1] and
    return an output of shape [... , `output_dim`, 1]
    (therefore supports batch processing).
    The added noise is provided by this class.
    """

    def __init__(self, alpha=1., beta=2., kappa=None,
                 sigma_noise=0., trainable_noise=False,
                 loss_samples=100, dtype='float64'):
        """
        Initialises the unscented transform based function.

        Args:
            alpha : Parameter for the unscented transform (default 1.)
            beta : Parameter for the unscented transform (default 2.)
            kappa : Parameter for the unscented transform
                (default 3 - input_dim)
            sigma_noise : Initial scale of the Gaussian noise (default 0.)
            trainable_noise : Boolean indicating if the noise scale
                is trainable.  (default True)
            loss_samples : Number of samples to use when estimating the
                loss function.  (default 100)
            dtype : TensorFlow type used for the function's variables
                and operations (default 'float64')
            name : Name of the function (default 'NNR')
            **kwargs : Supplementary arguments for the parent class.
        """

        self._loss_samples = loss_samples

        self._sigma_noise = tf.Variable(
            sigma_noise * tf.ones([self.output_dim], dtype=self._dtype),
            trainable_noise, name='sigma_noise')
        self._sigma_noise2 = tf.square(self._sigma_noise)
        self._added_noise_var = tf.diag(self._sigma_noise2)

        if kappa is None:
            kappa = 3 - self.input_dim

        lam = alpha ** 2 * (self.input_dim + kappa) - self.input_dim
        c = self.input_dim + lam

        self._sqrt_c = np.sqrt(c)

        self._weights_mean = np.ones((2 * self.input_dim + 1)) / c
        self._weights_mean[0] *= lam
        self._weights_mean[1:] /= 2
        self._weights_cov = np.copy(self._weights_mean)
        self._weights_cov[0] += 1 - alpha ** 2 + beta

    def variables_to_save(self):
        """
        Implementation of the function variables_to_save
        required by GDSM. (see template.py)
        """
        return [self._sigma_noise]

    def moments_to_points(self, mu, Sigma):
        """
        Transforms mean and covariance into sigma points.

        Args:
            mu : `Tensor` of shape [... , input_dim, 1]
            Sigma : `Tensor` of shape [... , input_dim, input_dim]

        Returns:
            A `Tensor` of shape [... , 2*input_dim + 1, input_dim, 1]
        """

        temp = self._sqrt_c * tf.cholesky(Sigma)
        points = tf.concat([mu, mu + temp, mu - temp], -1)

        return tf.expand_dims(tf.matrix_transpose(points), -1)

    def points_to_moments(self, points_in, points_out):
        """
        Transforms a set of transformed sigma points into an
        estimate of mean and covariance.

        Args:
            points_in : Sigma points before transformation
                as a `Tensor` of shape
                [... , 2*input_dim + 1, input_dim, 1]
            points_out : Sigma points after transformation
                as a `Tensor` of shape
                [... , 2*input_dim + 1, output_dim, 1]

        Returns:
            3 `Tensor`s representing the output mean,
            the output covariance and the covariance between
            the output and the input as :
            `Tensor` of shape [... , output_dim, 1]
            `Tensor` of shape [... , output_dim, output_dim]
            `Tensor` of shape [... , output_dim, input_dim]
        """

        points_in = tf.matrix_transpose(tf.squeeze(points_in, -1))
        points_out = tf.matrix_transpose(tf.squeeze(points_out, -1))

        m = tf.reduce_sum(points_out * self._weights_mean,
                          axis=-1, keep_dims=True)

        temp_out = points_out - m
        temp_out_weighted = temp_out * self._weights_cov
        temp_in = points_in - tf.reduce_sum(points_in * self._weights_mean,
                                            axis=-1, keep_dims=True)

        C = tf.matmul(temp_out_weighted, temp_out, transpose_b=True)
        C_oi = tf.matmul(temp_out_weighted, temp_in, transpose_b=True)

        return m, C, C_oi

    def propagate(self, mu, Sigma):
        """
        Implementation of the function propagate
        required by GDSM. (see template.py)
        """

        points_in = self.moments_to_points(mu, Sigma)
        points_out = self.apply_to(points_in)

        m, C, C_oi = self.points_to_moments(points_in, points_out)

        # Add noise
        C += self._added_noise_var

        return m, C, C_oi

    def loss_function(self, m_i, C_i, m_o, C_o=None, C_io=None):
        """
        Implementation of the function loss_function
        required by GDSM. (see template.py)
        """

        sample_shape = tf.concat([tf.shape(m_i)[:-1], [self._loss_samples]], 0)

        # Sample input
        chol = tf.cholesky(C_i)
        raw_sample = tf.random_normal(sample_shape, dtype=self._dtype)
        input_sample = tf.matmul(chol, raw_sample) + m_i
        input_sample = tf.expand_dims(tf.matrix_transpose(input_sample), -1)

        output_sample = self.apply_to(input_sample)
        output_sample = tf.matrix_transpose(tf.squeeze(output_sample, -1))

        if C_o is not None:
            temp = tf.matrix_triangular_solve(chol, C_io)
            m_o += tf.matmul(temp, raw_sample, transpose_a=True)
            C_o -= tf.matmul(temp, temp, transpose_a=True)

        temp = tf.reduce_mean(tf.square(output_sample - m_o), -1)
        if C_o is not None:
            temp += tf.matrix_diag_part(C_o)

        return temp / self._sigma_noise2 \
            + 2 * tf.log(tf.abs(self._sigma_noise))

    def sample(self, inputs):
        """
        Implementation of the function sample
        required by GDSM. (see template.py)
        """

        m = self.apply_to(inputs)

        return m + tf.random_normal(tf.shape(m), dtype=self._dtype) \
            * tf.expand_dims(self._sigma_noise, -1)
