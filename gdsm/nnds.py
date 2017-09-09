import tensorflow as tf

from .gdsm import GDSM
from .nnr import NNR
from .linear import Linear


class NNDS(GDSM):
    """
    Neural Network Dynamical System
    This class defines the dynamical system using GDSM as parent class.
    The dynamics consist in a neural network (see nnr) as transition function
    and a linear transformation as observation function (see linear).
    """

    def __init__(self, observations_dim,
                 state_dim,
                 state_scale=1.0,
                 state_noise=1e-2,
                 hidden_layers=1,
                 hidden_dim=None,
                 obs_scale=1.0,
                 obs_noise=1e-2,
                 trainable_dyn_noise=False,
                 trainable_obs_noise=False,
                 trainable_obs=True,
                 loss_samples=100,
                 dtype='float64',
                 name='NNDS',
                 **kwargs):
        """
        Initialises the NNDS model.

        Args:
            observations_dim : Dimension of the observations
            state_dim : Dimension of the hidden states
            state_scale : Scale of the states values (default 1.0)
            state_noise : Scale of the Gaussian noise added to
                the transition function (default 1e-2)
            hidden_layers : Number of hidden layers in the NNR (default 1)
            hidden_dim : Dimension of the hidden layer in in the NNR
                Can be an `Int` or a list of `Int`s (default None)
            obs_scale : Scale of the observations values (default 1.0)
            obs_noise : Scale of the Gaussian noise added to
                the observations (default 1e-2)
            trainable_dyn_noise : Boolean indicating if the states noise
                scale is trainable (default False)
            trainable_obs_noise : Boolean indicating if the observations noise
                scale is trainable (default False)
            trainable_obs : Boolean indicating if the complete
                observation function is trainable (default True)
            loss_samples : Number of samples to use when estimating
                the cost function for the NNR (default 100)
            dtype : TensorFlow type used for the model's variables
                and operations (default 'float64')
            name : Name of the model (default 'NNDS')
            **kwargs : Supplementary keyword arguments for the parent GDSM class
        """

        with tf.name_scope(name):
            # Build state transition
            transition_function = \
                NNR(state_dim, state_dim,
                    sigma_init=state_scale, sigma_noise=state_noise,
                    hidden_layers=hidden_layers, hidden_dim=hidden_dim,
                    trainable_noise=trainable_dyn_noise,
                    loss_samples=loss_samples,
                    dtype=dtype, name='TransitionFunction')

            # Build observation
            observation_function = \
                Linear(state_dim, observations_dim,
                       sigma_init=obs_scale, sigma_noise=obs_noise,
                       trainable_noise=trainable_obs_noise,
                       trainable=trainable_obs,
                       dtype=dtype, name='ObservationFunction')

            super().__init__(transition_function, observation_function,
                             dtype=dtype, name=name, **kwargs)


Model = NNDS
