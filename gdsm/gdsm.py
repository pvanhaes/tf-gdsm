import tensorflow as tf
import numpy as np


class GDSM:
    """
    Gaussian shaped Dynamical System Model
    This class implements a dynamical system on which you can
    perform parameter estimation using the EM algorithm.

    The E-step is performed using the Kalman filter.
    You can provided your own M-step applied to the `loss_function`
    made available.

    You have to provide two objects to define the dynamics:
    a transition function and an observation function.
    See template.py for details on the required operations to
    give to your dynamics objects.
    """

    def __init__(self, transition_function,
                 observation_function,
                 missing_obs=False,
                 default_m_step=True,
                 dtype='float64', name='GDSM'):
        """
        Initialises the GDSM model given a transition and observation function.

        Args:
            transition_function : Object following guidelines from template.py
            observation_function : Object following guidelines from template.py
            missing_obs : Boolean indicating if the model should handle missing
                values inside the sequences represented with NaNs,
                can make the model a bit slower (default False)
            default_m_step : Boolean indicating if the GDSM object should create its
                own M-step operation (currently a simple
                GradientDescentOptimizer with learning rate 1e-5)
                (default True)
            dtype : TensorFlow type used for the model's variables
                and operations (default 'float64')
            name : Name of the model (default 'GDSM')
        """

        self._f = transition_function
        self._g = observation_function

        self._use_mask = missing_obs
        self._default_m_step = default_m_step
        self._dtype = dtype

        self._state_dim = self._f.input_dim
        self._observations_dim = self._g.output_dim

        # Build the graph
        with tf.name_scope(name):
            self._build_graph()

    def _build_graph(self):

        # Initial state variables
        self._mu0 = tf.Variable(
            tf.zeros([self._state_dim, 1], dtype=self._dtype),
            trainable=False, name='mu0')
        self._Sigma0 = tf.Variable(tf.eye(self._state_dim, dtype=self._dtype),
                                   trainable=False, name='Sigma0')

        # Placeholders
        self._obs_pl = tf.placeholder(
            shape=[None, None, self._observations_dim, 1],
            dtype=self._dtype, name='observations')

        with tf.name_scope('Smoothing'):

            self._obs_var = tf.Variable(self._obs_pl,
                                        validate_shape=False,
                                        trainable=False,
                                        name='obs_var',
                                        collections=[
                                            tf.GraphKeys.LOCAL_VARIABLES])
            self._obs_var.set_shape(self._obs_pl.get_shape())

            observations = self._obs_var
            if self._use_mask:
                mask = tf.logical_not(tf.is_nan(observations))

                observations = tf.where(mask,
                                        observations,
                                        tf.zeros_like(observations))

                mask = tf.cast(mask, self._dtype)

                self._C_obs_mask = mask * tf.matrix_transpose(mask)
                self._obs_mask = tf.squeeze(mask, -1)
                self._f_mask = tf.reduce_max(self._obs_mask[1:], -1,
                                             keep_dims=True)

            self._observations = observations

            m, C, C_cross = self._smooth()

            self._m = tf.Variable(m, validate_shape=False,
                                  trainable=False, name='m',
                                  collections=[
                                      tf.GraphKeys.LOCAL_VARIABLES])
            self._m.set_shape(m.get_shape())

            self._C = tf.Variable(C, validate_shape=False,
                                  trainable=False, name='C',
                                  collections=[
                                      tf.GraphKeys.LOCAL_VARIABLES])
            self._C.set_shape(C.get_shape())

            self._C_cross = tf.Variable(C_cross, validate_shape=False,
                                        trainable=False, name='C_cross',
                                        collections=[
                                            tf.GraphKeys.LOCAL_VARIABLES])
            self._C_cross.set_shape(C_cross.get_shape())

        with tf.name_scope('Training') as scope:
            # Build training ops
            self._e_step = [self._m.initializer,
                            self._C.initializer,
                            self._C_cross.initializer]
            # Build loss
            self._build_loss()

            if self._default_m_step:
                optimizer = tf.train.GradientDescentOptimizer(1e-5)
                self._m_step = optimizer.minimize(self.loss)

            self._batch_ratio = tf.placeholder_with_default(
                tf.zeros([], dtype=self._dtype), shape=[])

            # mu0 and Sigma0 update
            m0 = self._m[0]
            batch_mu0 = tf.reduce_mean(m0, 0)

            temp1 = m0 - batch_mu0
            temp1 = tf.matmul(temp1, temp1, transpose_b=True)

            temp2 = batch_mu0 - self._mu0
            temp2 = tf.matmul(temp2, temp2, transpose_b=True)

            batch_Sigma0 = tf.reduce_mean(self._C[0] + temp1, 0)

            self._update_mu0 = self._mu0.assign(
                (1. - self._batch_ratio) * self._mu0
                + self._batch_ratio * batch_mu0)
            self._update_Sigma0 = self._Sigma0.assign(
                (1. - self._batch_ratio) * self._Sigma0
                + self._batch_ratio * (
                    batch_Sigma0
                    + (1. - self._batch_ratio) * temp2))

        with tf.name_scope('Prediction'):
            # Build prediction ops
            self._predict_length = tf.placeholder(shape=[], dtype='int32')
            self._concatenate_pred = tf.placeholder(shape=[], dtype='bool')
            self._build_predicted_output()

    def variables_to_save(self):
        """
        Returns a list of the TensorFlow variables contained in this model.
        It also fetches the variables of the transition and observation function
        by calling their method of the same name.
        """
        return [self._mu0, self._Sigma0] \
            + self._f.variables_to_save() \
            + self._g.variables_to_save()

    def set_batch_tensor(self, tensor):
        """
        Sets a `Tensor` used as input for training.
        Can be a `dequeue` op for example when working with
        a queue.

        Args:
            tensor : A `Tensor` of shape [T, ?, input_dim]
                where T is the length of the sequences.
                '?' means that this dimension may not be used.
        """
        tensor = tf.reshape(
            tensor,
            [tf.shape(tensor)[0], -1, self._observations_dim, 1])

        self._read_obs = tf.assign(self._obs_var, tensor, validate_shape=False)

    def set_m_step(self, m_step):
        """
        Sets the m_step operation called during training.
        This should be the result of an optimizer minimizing the `loss`.

        Args:
            m_step : the m_step operation
        """
        self._m_step = m_step

    def train_on_batch(self, observations=None,
                       batch_ratio=1.0, em_steps=1, m_steps=1,
                       sess=None, verbose=False):
        """
        Trains the GDSM model using the EM algorithm.

        Args:
            observations : An array containing the observations to use
                for training. Should be of shape [T, ?, input_dim]
                '?' means that this dimension may not be used.
                If None the model uses the Tensor provided by
                calling the function `set_batch_tensor`.
                (default None)
            batch_ratio : Proportion of data inside the batch with respect to
                the complete dataset. This is used to update the
                initial state prior distribution accordingly.
                (default 1.0)
            em_steps : Number of EM steps to perform on the batch.
                (default 1)
            em_steps : Number of M steps to perform for each EM step.
                (default 1)
            sess : The TensorFlow session to use for run the operations.
                Uses default session from TensorFlow if not set.
            verbose : Boolean indicating if informations should be printed
                during training.
        """

        if sess is None:
            sess = tf.get_default_session()

        # Read batch data
        if observations is None:
            sess.run(self._read_obs)
        else:
            observations = np.reshape(
                observations,
                [len(observations), -1, self._observations_dim, 1])

            sess.run(self._obs_var.initializer,
                     feed_dict={self._obs_pl: observations})

        loss_history = []

        for i in range(em_steps):
            sess.run(self._e_step)

            for _ in range(m_steps):
                _, loss_val = sess.run([self._m_step, self.loss])
                loss_history.append(loss_val)

            if m_steps > 0:
                sess.run([self._update_mu0, self._update_Sigma0],
                         feed_dict={self._batch_ratio: batch_ratio})

            if verbose:
                print('Done {} EM steps. Loss={}     '.format(
                    i + 1, loss_val), end='\r')

        return loss_history

    def predict(self, observations, total_length=None,
                concatenate_pred=True, sess=None):
        """
        Tries to increase the length of given sequences by making predictions.
        It can also be used to fill holes within the time series.

        Args:
            observations : An array containing the observations to use.
                Should be of shape [T, ?, input_dim]
                '?' means that this dimension may not be used.
            total_length : Final length of the returned sequences
                including predictions.
                Does increase the initial length by default.
            concatenate_pred : Boolean indicating if the returned sequences
                should include the initial time steps from
                the `observations` argument.
                (default True)
            sess : The TensorFlow session to use for run the operations.
                Uses default session from TensorFlow if not set.

        Returns:
            Two `Tensors` representing the mean and covariance of the
            resulting predictions.
            Outputs shapes are [total_length, ?, output_dim, 1]
            [total_length, ?, output_dim, output_dim].
        """

        if sess is None:
            sess = tf.get_default_session()

        if total_length is None:
            total_length = len(observations)

        observations = np.reshape(
            observations,
            [len(observations), -1, self._observations_dim, 1])

        sess.run(self._obs_var.initializer,
                 feed_dict={self._obs_pl: observations})
        sess.run(self._e_step)

        return sess.run(self._predict_op,
                        feed_dict={self._predict_length: total_length,
                                   self._concatenate_pred: concatenate_pred})

    def _smooth(self):

        m_s_list, C_s_list, C_cross_list = \
            self._backward_pass(*self._forward_pass())

        return m_s_list.stack(), C_s_list.stack(), C_cross_list.stack()

    def _filter(self):

        _, _, m_e_list, C_e_list, _ = self._forward_pass()

        return m_e_list.stack(), C_e_list.stack()

    def _forward_pass(self):
        # To perform batch processing the first dimension must be T

        with tf.name_scope('ForwardPass'):
            T = tf.shape(self._observations)[0]
            batch_shape = tf.shape(self._observations)[1:-2]

            m_p_list = tf.TensorArray(self._dtype, T)
            C_p_list = tf.TensorArray(self._dtype, T)
            m_e_list = tf.TensorArray(self._dtype, T)
            C_e_list = tf.TensorArray(self._dtype, T)
            C_pe_list = tf.TensorArray(self._dtype, T - 1)

            def observation_update(i, m_p, C_p):

                # Observation update
                m_obs, C_obs, C_obsp = self._g.propagate(m_p, C_p)

                # Use Gaussian conditioning to find filtered mean and cov
                if self._use_mask:
                    C_obs *= self._C_obs_mask[i]
                    temp = tf.matrix_solve_ls(C_obs, C_obsp, fast=False)

                    m_e = m_p + tf.matmul(temp, self._observations[i] - m_obs,
                                          transpose_a=True)
                    C_e = C_p - tf.matmul(temp, C_obsp, transpose_a=True)
                else:
                    chol = tf.cholesky(C_obs)
                    temp = tf.matrix_triangular_solve(chol, C_obsp)

                    m_e = m_p + tf.matmul(temp,
                                          tf.matrix_triangular_solve(
                                              chol, self._observations[i] - m_obs),
                                          transpose_a=True)
                    C_e = C_p - tf.matmul(temp, temp, transpose_a=True)

                return m_e, C_e

            # Write first state
            m_p, C_p = self._mu0 * tf.ones(
                shape=tf.concat([batch_shape, [self._state_dim, 1]], 0),
                dtype=self._dtype), \
                self._Sigma0 * tf.ones(
                shape=tf.concat(
                    [batch_shape, [self._state_dim, self._state_dim]], 0),
                dtype=self._dtype)

            m_e, C_e = observation_update(0, m_p, C_p)

            m_p_list = m_p_list.write(0, m_p)
            C_p_list = C_p_list.write(0, C_p)
            m_e_list = m_e_list.write(0, m_e)
            C_e_list = C_e_list.write(0, C_e)

            def body(i, m_e_prev, C_e_prev, m_p_list, C_p_list,
                     m_e_list, C_e_list, C_pe_list):

                # Time update
                m_p, C_p, C_pe = self._f.propagate(m_e_prev, C_e_prev)

                # Observation update
                m_e, C_e = observation_update(i, m_p, C_p)

                m_p_list = m_p_list.write(i, m_p)
                C_p_list = C_p_list.write(i, C_p)
                m_e_list = m_e_list.write(i, m_e)
                C_e_list = C_e_list.write(i, C_e)
                C_pe_list = C_pe_list.write(i - 1, C_pe)

                return i + 1, m_e, C_e, m_p_list, C_p_list, \
                    m_e_list, C_e_list, C_pe_list

            _, _, _, m_p_list, C_p_list, m_e_list, C_e_list, C_pe_list = \
                tf.while_loop(lambda i, *_: i < T, body,
                              [1, m_e, C_e,
                               m_p_list, C_p_list,
                               m_e_list, C_e_list, C_pe_list],
                              back_prop=False)

            return m_p_list, C_p_list, m_e_list, C_e_list, C_pe_list

    def _backward_pass(self, m_p_list, C_p_list,
                       m_e_list, C_e_list, C_pe_list):

        with tf.name_scope('BackwardPass'):
            T = m_e_list.size()

            m_s_list = tf.TensorArray(self._dtype, T)
            C_s_list = tf.TensorArray(self._dtype, T)
            C_cross_list = tf.TensorArray(self._dtype, T - 1)

            # Write last state
            m_s = m_e_list.read(T - 1)
            C_s = C_e_list.read(T - 1)
            m_s_list = m_s_list.write(T - 1, m_s)
            C_s_list = C_s_list.write(T - 1, C_s)

            def body(i, m_s_prev, C_s_prev, m_s_list, C_s_list, C_cross_list):

                C_p = C_p_list.read(i + 1)

                chol = tf.cholesky(C_p)
                temp = tf.matrix_triangular_solve(chol, C_pe_list.read(i))

                m_s = m_e_list.read(i) \
                    + tf.matmul(
                        temp,
                        tf.matrix_triangular_solve(
                            chol,
                            m_s_prev - m_p_list.read(i + 1)),
                        transpose_a=True)

                C_s = C_e_list.read(i) \
                    + tf.matmul(temp,
                                tf.matrix_triangular_solve(
                                    chol,
                                    tf.matmul(
                                        tf.matrix_triangular_solve(
                                            chol,
                                            C_s_prev - C_p),
                                        temp,
                                        transpose_a=True)),
                                transpose_a=True)

                C_cross = tf.matmul(temp,
                                    tf.matrix_triangular_solve(chol, C_s_prev),
                                    transpose_a=True)

                m_s_list = m_s_list.write(i, m_s)
                C_s_list = C_s_list.write(i, C_s)
                C_cross_list = C_cross_list.write(i, C_cross)

                return i - 1, m_s, C_s, m_s_list, C_s_list, C_cross_list

            _, _, _, m_s_list, C_s_list, C_cross_list = \
                tf.while_loop(lambda i, *_: i >= 0, body,
                              [T - 2, m_s, C_s,
                               m_s_list, C_s_list, C_cross_list],
                              back_prop=False)

            return m_s_list, C_s_list, C_cross_list

    def _build_loss(self):

        with tf.name_scope('Loss'):

            f_cost = self._f.loss_function(
                self._m[:-1], self._C[:-1],
                self._m[1:], self._C[1:], self._C_cross)
            g_cost = self._g.loss_function(
                self._m, self._C, self._observations)

            if self._use_mask:
                f_cost *= self._f_mask
                g_cost *= self._obs_mask

            self.loss = tf.reduce_mean(f_cost) + tf.reduce_mean(g_cost)

    def _build_predicted_output(self):

        prediction_length = self._predict_length - \
            tf.shape(self._observations)[0]
        m_begin, C_begin = self._m, self._C

        def body(i, m_prev, C_prev, m_pred_list, C_pred_list):

            m, C, _ = self._f.propagate(m_prev, C_prev)

            m_pred_list = m_pred_list.write(i, m)
            C_pred_list = C_pred_list.write(i, C)

            return i + 1, m, C, m_pred_list, C_pred_list

        _, _, _, m_pred_list, C_pred_list = \
            tf.while_loop(lambda i, *_: i < prediction_length, body,
                          [0, m_begin[-1],
                              C_begin[-1],
                           tf.TensorArray(self._dtype, prediction_length),
                           tf.TensorArray(self._dtype, prediction_length)],
                          back_prop=False)

        m_total = m_pred_list.stack()
        C_total = C_pred_list.stack()

        m_total, C_total = tf.cond(self._concatenate_pred,
                                   lambda: (tf.concat([m_begin, m_total], 0),
                                            tf.concat([C_begin, C_total], 0)),
                                   lambda: (m_total, C_total))

        m_obs_total, C_obs_total, _ = self._g.propagate(m_total, C_total)

        self._predict_op = [m_obs_total, C_obs_total]
