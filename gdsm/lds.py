import tensorflow as tf

from .gdsm import GDSM
from .linear import Linear


class LDS(GDSM):
    """
    Linear System
    This class defines the dynamical system using GDSM as parent class.
    The dynamics consist in linear transformations for both the transition and
    the observation function.
    """

    def __init__(self, observations_dim,
                 state_dim,
                 state_scale=1.0,
                 state_noise=1e-2,
                 obs_scale=1.0,
                 obs_noise=1e-2,
                 trainable_dyn_noise=False,
                 trainable_obs_noise=False,
                 trainable_obs=True,
                 dtype='float64',
                 name='LDS',
                 **kwargs):
        """
        Initialises the NNDS model.

        Args:
            observations_dim : Dimension of the observations
            state_dim : Dimension of the hidden states
            state_scale : Scale of the states values (default 1.0)
            state_noise : Scale of the Gaussian noise added to
                the transition function (default 1e-2)
            obs_scale : Scale of the observations values (default 1.0)
            obs_noise : Scale of the Gaussian noise added to
                the observations (default 1e-2)
            trainable_dyn_noise : Boolean indicating if the states noise
                scale is trainable (default False)
            trainable_obs_noise : Boolean indicating if the observations noise
                scale is trainable (default False)
            trainable_obs : Boolean indicating if the complete
                observation function is trainable (default True)
            dtype : TensorFlow type used for the model's variables
                and operations (default 'float64')
            name : Name of the model (default 'LDS')
            **kwargs : Supplementary keyword arguments for the parent GDSM class
        """

        with tf.name_scope(name):
            # Build state transition
            transition_function = \
                Linear(state_dim, state_dim,
                       sigma_init=state_scale, sigma_noise=state_noise,
                       trainable_noise=trainable_dyn_noise,
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


Model = LDS
