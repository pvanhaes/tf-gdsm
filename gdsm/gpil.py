import tensorflow as tf

from .gdsm import GDSM
from .gp import GP
from .linear import Linear


class GPIL(GDSM):
    """
    Gaussian Process Inference and Learning (see Turner et al. 2010
    "State-space inference and learning with Gaussian processes")
    This class defines the dynamical system using GDSM as parent class.
    The dynamics consist in Gaussian processes (see gp) as transition
    and observation function.
    """

    def __init__(self, observations_dim,
                 state_dim,
                 n=16,
                 state_set_size=None,
                 state_scale=0.1,
                 state_lengthscale=1.0,
                 state_noise=1e-2,
                 smooth_states=True,
                 obs_set_size=None,
                 obs_scale=1.0,
                 obs_lengthscale=1.0,
                 obs_noise=1e-2,
                 gp_obs=True,
                 trainable_dyn_noise=False,
                 trainable_obs_noise=False,
                 trainable_obs=True,
                 train_pseudo_dataset_only=True,
                 loss_samples=0,
                 dtype='float64',
                 name='GPIL',
                 **kwargs):
        """
        Initialises the NNDS model.

        Args:
            observations_dim : Dimension of the observations
            state_dim : Dimension of the hidden states
            n : Size of the pseudo dataset for both functions, when not `None`
                this parameter takes precedence over `state_set_size`
                and `obs_set_size`.  (default 16)
            state_set_size : Size of the pseudo dataset of the transition function
                (default None)
            state_scale : Scale of the states values (default 0.1)
            state_noise : Scale of the Gaussian noise added to
                the transition function (default 1e-2)
            smooth_states : Boolean indicating if the transition function
                has the identity as mean or zero instead. (default True)
            obs_set_size : Size of the pseudo dataset of the observation function
                (default None)
            obs_scale : Scale of the observations values (default 1.0)
            obs_noise : Scale of the Gaussian noise added to
                the observations (default 1e-2)
            gp_obs : Boolean indicating whether to use a GP as observation
                function or not (default False)
            trainable_nyn_noise : Boolean indicating if the states noise
                scale is trainable (default False)
            trainable_obs_noise : Boolean indicating if the observations noise
                scale is trainable (default False)
            trainable_obs : Boolean indicating if the complete
                observation function is trainable (default True)
            loss_samples : Number of samples to use when estimating
                the cost function, 0 means using the approximation
                (default 0)
            dtype : TensorFlow type used for the model's variables
                and operations (default 'float64')
            name : Name of the model (default 'GPIL')
            **kwargs : Supplementary keyword arguments for the parent GDSM class
        """

        if state_set_size is None:
            state_set_size = n
        if obs_set_size is None:
            obs_set_size = n

        with tf.name_scope(name):
            # Build state transition GP
            transition_function = \
                GP(state_dim, state_dim, id_mean=smooth_states,
                    n=state_set_size,
                    sigma_init=state_scale, lengthscale_init=state_lengthscale,
                    sigma_noise=state_noise,
                    trainable_noise=trainable_dyn_noise,
                    train_pseudo_dataset_only=train_pseudo_dataset_only,
                    loss_samples=loss_samples,
                    dtype=dtype, name='TransitionFunction')

            # Build observation GP
            if gp_obs:
                observation_function = \
                    GP(state_dim, observations_dim, id_mean=False,
                        n=obs_set_size,
                        sigma_init=obs_scale, lengthscale_init=obs_lengthscale,
                        sigma_noise=obs_noise,
                        trainable_noise=trainable_obs_noise,
                        train_pseudo_dataset_only=train_pseudo_dataset_only,
                        trainable=trainable_obs, loss_samples=loss_samples,
                        dtype=dtype, name='ObservationFunction')
            else:
                observation_function = \
                    Linear(state_dim, observations_dim,
                           sigma_init=obs_scale, sigma_noise=obs_noise,
                           trainable_noise=trainable_obs_noise,
                           trainable=trainable_obs,
                           dtype=dtype, name='ObservationFunction')

            super().__init__(transition_function, observation_function,
                             dtype=dtype, name=name, **kwargs)


Model = GPIL
