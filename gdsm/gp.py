import tensorflow as tf

from .utils import my_reduce_prod


class GP:
    """
    Implementation of a parametrized Gaussian process with RBF kernel.
    See definition in Turner et al. 2010
    "State-space inference and learning with Gaussian processes".
    This class is made to be used inside the GDSM implementation.
    """

    def __init__(self, input_dim, output_dim=None, id_mean=False,
                 n=10, sigma_init=1., lengthscale_init=1., sigma_noise=0.,
                 trainable_noise=False, train_pseudo_dataset_only=False,
                 trainable=True, loss_samples=0,
                 dtype='float64', name='GP'):
        """Initializes the GP.

        Args:
            input_dim : Dimension of the input
            output_dim : Dimension of the output
            id_mean : Boolean indicating if the mean function the identity,
                if True forces output_dim to be equal to input_dim.
                (default False)
            n : Size of the pseudo training set (default 10)
            sigma_init : Initial scale, at initialisation, of the transformation.
                Usually it can be set to 1 for the transition function,
                for the observation it should be the standard deviation of your
                observations to simplify training. (default 1.)
            lengthscale_init : Initial lengthscale of the GP. (default 1.)
            sigma_noise : Initial scale of the Gaussian noise added to the
                result of the GP (default 0.)
            trainable_noise : Boolean indicating if the noise scale is trainable.
                (default True)
            train_pseudo_dataset_only : Boolean indicating if only the training set
                should be trained and not the other params.
                like sigma, lengthscale and sigma_noise.
            trainable : Boolean indicating if the complete transformation is trainable.
                (default True)
            loss_samples : Number of samples to use when estimating the loss function.
                (default 100)
            dtype : TensorFlow type used for the function's variables
                and operations (default 'float64')
            name : Name of the function (default 'GP')
        """

        self._dtype = dtype
        self._id_mean = id_mean
        self.input_dim = input_dim
        self.output_dim = output_dim if not self._id_mean else self.input_dim
        self._n = n
        self._loss_samples = loss_samples

        with tf.name_scope(name):
            # Hyperparameters
            self._sigma = tf.Variable(
                sigma_init, trainable and not train_pseudo_dataset_only,
                name='sigma', dtype=self._dtype)
            self._sigma2 = tf.square(self._sigma)

            self._lengthscale = tf.Variable(
                lengthscale_init, trainable and not train_pseudo_dataset_only,
                name='lengthscale', dtype=self._dtype)
            self._lengthscale2 = tf.square(self._lengthscale)

            self._sigma_noise = tf.Variable(
                sigma_noise,
                trainable and trainable_noise and not train_pseudo_dataset_only,
                name='sigma_noise', dtype=self._dtype)
            self._sigma_noise2 = tf.square(self._sigma_noise)

            # Inputs / outputs variables
            inputs_init = tf.random_normal(
                [self._n, self.input_dim], dtype=self._dtype)
            self._inputs = tf.Variable(inputs_init, trainable, name='inputs')

            outputs_init = tf.random_normal([self._n, self.output_dim],
                                            stddev=sigma_init,
                                            dtype=self._dtype)
            if self._id_mean:
                outputs_init += inputs_init
            self._outputs = tf.Variable(
                outputs_init, trainable, name='outputs')

            # Precomputed terms
            self._Lambda = tf.eye(self.input_dim, dtype=self._dtype) \
                * self._lengthscale2
            self._Iout = tf.eye(self.output_dim, dtype=self._dtype)

            # Compute covariance matrix
            X = self._inputs / self._lengthscale
            Xs = tf.reduce_sum(tf.square(X), 1)
            square_dist = - 2. * tf.matmul(X, X, transpose_b=True) \
                + tf.expand_dims(Xs, 0) + tf.expand_dims(Xs, 1)
            K = self._sigma2 * tf.exp(-square_dist / 2.) \
                + self._sigma_noise2 * tf.eye(self._n, dtype=self._dtype)

            self._Kchol = tf.cholesky(K)

            # Compute beta
            self._beta = tf.cholesky_solve(self._Kchol,
                                           self._outputs if not self._id_mean
                                           else self._outputs - self._inputs)

            self._inputsT = tf.transpose(self._inputs)

            # Precompute L
            temp = tf.abs(self._lengthscale) ** self.input_dim
            self._lprecomp = self._sigma2 * temp
            self._Lprecomp1 = tf.reshape(
                (tf.expand_dims(self._inputsT, -1)
                 + tf.expand_dims(self._inputsT, -2)) / 2.,
                [self.input_dim, -1])
            self._Lprecomp2 = tf.square(
                self._sigma2) * temp * tf.exp(-square_dist / 4.)

    def variables_to_save(self):
        """
        Implementation of the function variables_to_save
        required by GDSM. (see template.py)
        """
        return [self._inputs,
                self._outputs,
                self._sigma,
                self._lengthscale,
                self._sigma_noise]

    def propagate(self, mu, Sigma):
        """
        Implementation of the function propagate
        required by GDSM. (see template.py)
        """

        do_batch = mu.get_shape().ndims != 2
        batch_shape = tf.shape(mu)[:-2]

        l, L, H = self._propagation_terms(mu, Sigma)

        # Compute predicted mean
        l_rank2 = l if not do_batch else tf.reshape(l, [-1, self._n])
        m = tf.matmul(l_rank2, self._beta)
        if do_batch:
            m = tf.reshape(m, tf.concat([batch_shape,
                                         [self.output_dim, 1]], 0))
        else:
            m = tf.transpose(m)

        L_rank2 = L if not do_batch else tf.reshape(L, [-1, self._n])

        # Compute predicted output variance
        temp = tf.matmul(L_rank2, self._beta)
        if do_batch:
            temp = tf.reshape(temp, [-1, self._n, self.output_dim])
            temp = tf.matrix_transpose(temp)
            temp = tf.reshape(temp, [-1, self._n])
            temp = tf.matmul(temp, self._beta)
            temp = tf.reshape(
                temp,
                tf.concat([batch_shape, [self.output_dim, self.output_dim]],
                          0))
            L_rank2 = tf.transpose(L_rank2)
        else:
            temp = tf.matmul(temp, self._beta, transpose_a=True)

        C = temp - tf.matmul(m, m, transpose_b=True)

        temp = tf.cholesky_solve(self._Kchol, L_rank2)
        if do_batch:
            temp = tf.reshape(
                tf.transpose(temp),
                tf.concat([batch_shape,
                           [1, 1, self._n, self._n]], 0))

        C += self._Iout * \
            (self._sigma2 + self._sigma_noise2 - tf.trace(temp))

        # Compute input/output covariance
        C_oi = tf.matmul(l * tf.transpose(self._beta),
                         H - mu, transpose_b=True)

        if self._id_mean:
            m += mu
            C += Sigma + C_oi + tf.matrix_transpose(C_oi)
            C_oi += Sigma

        return m, C, C_oi

    def loss_function(self, m_i, C_i, m_o, C_o=None, C_io=None):
        """
        Implementation of the function loss_function
        required by GDSM. (see template.py)
        """
        # C_o == None : output is deterministic

        if self._loss_samples > 0:
            return self._loss_function_sampled(m_i, C_i, m_o, C_o, C_io)

        do_batch = m_i.get_shape().ndims != 2
        batch_shape = tf.shape(m_i)[:-2]

        l, L, H = self._propagation_terms(m_i, C_i)

        L_rank2 = L if not do_batch else \
            tf.reshape(L, [-1, self._n])

        temp = tf.matmul(L_rank2, self._beta)
        if do_batch:
            temp = tf.reshape(
                temp,
                tf.concat([batch_shape, [self._n, self.output_dim]], 0))
            L_rank2 = tf.transpose(L_rank2)
        temp = tf.reduce_sum(temp * self._beta, -2)

        cost = tf.squeeze(tf.square(m_o), -1) + temp

        temp = m_o
        if C_o is not None:
            chol = tf.cholesky(C_i)
            temp1 = tf.matrix_triangular_solve(chol, C_io)
            temp2 = tf.matrix_triangular_solve(chol, H - m_i)
            temp += tf.matmul(temp1, temp2, transpose_a=True)

            cost += tf.matrix_diag_part(C_o)

            if self._id_mean:
                cost -= 2 * tf.matrix_diag_part(C_io)

        if self._id_mean:
            temp -= H
            cost += tf.squeeze(tf.square(m_i), -1) + tf.matrix_diag_part(C_i) \
                - 2 * tf.squeeze(m_i * m_o, -1)

        cost -= 2 * tf.reduce_sum(l * (temp * tf.transpose(self._beta)), -1)

        temp = tf.cholesky_solve(self._Kchol, L_rank2)
        if do_batch:
            temp = tf.reshape(
                tf.transpose(temp),
                tf.concat([batch_shape, [1, self._n, self._n]], 0))

        E_sigma2 = self._sigma2 + self._sigma_noise2 - tf.trace(temp)

        return cost / E_sigma2 + tf.log(E_sigma2)

    def sample(self, inputs):
        """
        Implementation of the function sample
        required by GDSM. (see template.py)
        """

        batch_shape = tf.shape(inputs)[:-2]
        outputs_shape = tf.concat([batch_shape, [self.output_dim, 1]], 0)

        Ksample = self._sigma2 * tf.exp(- tf.reduce_sum(
            tf.square(inputs - self._inputsT), -2, keep_dims=True)
            / (2 * self._lengthscale2))

        temp = tf.reshape(Ksample, [-1, self._n])
        m = tf.reshape(tf.matmul(temp, self._beta), outputs_shape)

        temp = tf.matrix_triangular_solve(self._Kchol, tf.transpose(temp))
        temp = tf.reshape(tf.transpose(temp), tf.shape(Ksample))

        sigma2 = self._sigma2 + self._sigma_noise2 \
            - tf.reduce_sum(tf.square(temp), -1, keep_dims=True)

        # Sample
        sample = m + tf.random_normal(outputs_shape, dtype=self._dtype) \
            * tf.sqrt(tf.maximum(sigma2, 0))

        if self._id_mean:
            sample += inputs

        return sample

    def _loss_function_sampled(self, m_i, C_i, m_o, C_o=None, C_io=None):
        """
        Sampled version of the loss_function, used when loss_samples > 0.
        """

        do_batch = m_i.get_shape().ndims != 2
        sample_shape = tf.concat([tf.shape(m_i)[:-1], [self._loss_samples]], 0)
        batch_shape = sample_shape[:-2]

        # Sample input
        chol = tf.cholesky(C_i)
        raw_sample = tf.random_normal(sample_shape, dtype=self._dtype)
        input_sample = tf.matmul(chol, raw_sample) + m_i
        temp = tf.matrix_transpose(input_sample) / self._lengthscale

        # Compute mean for samples
        Ksample = self._sigma2 * tf.exp(- tf.reduce_sum(
            tf.square(self._inputs / self._lengthscale
                      - tf.expand_dims(temp, -2)), -1) / 2)
        if do_batch:
            Ksample = tf.reshape(Ksample, [-1, self._n])
        temp = tf.matmul(Ksample, self._beta)
        if do_batch:
            temp = tf.reshape(
                temp, tf.concat(
                    [batch_shape, [self._loss_samples, self.output_dim]], 0))
        mean_sample = tf.matrix_transpose(temp)
        if self._id_mean:
            mean_sample += input_sample

        # Compute variance for samples
        temp = tf.matrix_triangular_solve(
            self._Kchol,
            tf.transpose(Ksample))
        temp = tf.transpose(temp)
        if do_batch:
            temp = tf.reshape(
                temp, tf.concat(
                    [batch_shape, [self._loss_samples, self._n]], 0))
        temp = tf.reduce_sum(tf.square(temp), -1, keep_dims=True)
        sigma2 = tf.matrix_transpose(
            self._sigma2 + self._sigma_noise2 - temp)

        if C_o is not None:
            temp = tf.matrix_triangular_solve(chol, C_io)
            m_o += tf.matmul(temp, raw_sample, transpose_a=True)
            C_o -= tf.matmul(temp, temp, transpose_a=True)

        temp = tf.square(mean_sample - m_o)
        if C_o is not None:
            temp += tf.expand_dims(tf.matrix_diag_part(C_o), -1)

        return tf.reduce_mean(tf.log(sigma2) + temp / sigma2, axis=-1)

    def _propagation_terms(self, mu, Sigma):
        """
        Computes the intermediate results needed for propagation and loss_function.

        Supports batch processing.

        Args:
            mu : `Tensor` of shape [... , 1, input_dim]
            Sigma : `Tensor` of shape [... , input_dim, input_dim]

        Returns:
            l : `Tensor` of shape [... , 1, n]
            L : `Tensor` of shape [... , n, n]
            H : `Tensor` of shape [... , input_dim, n]
        """
        do_batch = mu.get_shape().ndims != 2
        batch_shape = tf.shape(mu)[:-2]

        diffs = self._inputsT - mu
        chol1 = tf.cholesky(Sigma + self._Lambda)
        temp1 = tf.matrix_triangular_solve(chol1, diffs)

        # Hack to avoid error when reducing on axis -1 (within gradient?..)
        temp = tf.matrix_diag_part(chol1)
        temp2 = my_reduce_prod(temp, -1, keep_dims=True)

        l = tf.exp(-tf.reduce_sum(tf.square(temp1), -2) / 2.) \
            / temp2 * self._lprecomp
        l = tf.expand_dims(l, -2)

        diffs = self._Lprecomp1 - mu
        chol2 = tf.cholesky(Sigma * 2 + self._Lambda)
        temp1 = tf.matrix_triangular_solve(chol2, diffs)

        # Hack to avoid error when reducing on axis -1 (within gradient?..)
        temp = tf.matrix_diag_part(chol2)
        temp2 = tf.expand_dims(my_reduce_prod(temp, -1, keep_dims=True), -1)

        # Compute L
        temp_shape = self._Lprecomp2.get_shape() if not do_batch else \
            tf.concat([batch_shape, tf.shape(self._Lprecomp2)], 0)

        L = tf.reshape(tf.exp(-tf.reduce_sum(tf.square(temp1), - 2)),
                       temp_shape) \
            / temp2 * self._Lprecomp2

        temp = Sigma if not do_batch else tf.reshape(
            Sigma, [-1, self.input_dim])
        temp = tf.matmul(temp, self._inputsT)
        if do_batch:
            temp = tf.reshape(
                temp,
                tf.concat([batch_shape, [self.input_dim, self._n]], 0))

        H = tf.cholesky_solve(chol1, temp + self._lengthscale2 * mu)

        return l, L, H
