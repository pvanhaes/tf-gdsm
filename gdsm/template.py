import tensorflow as tf


class Template:
    """
    Example template class showing which functions to implement
    for use in the GDSM model.
    """

    def __init__(self):
        """
        This class is empty.
        """
        pass

    def variables_to_save(self):
        """
        Returns a list of the variables to save in order to
        be able to recreate the function to identical.
        """
        pass

    def propagate():
        """
        Propagates an uncertain Gaussian input through the function.
        Must support batch processing.

        Args:
            mu : Mean of the input as `Tensor` of shape
                [... , input_dim, 1]
            Sigma : Covariance matrix of the input as `Tensor` of shape
                [... , input_dim, input_dim]

        Returns:
            3 `Tensor`s representing the output mean,
            the output covariance and the covariance between
            the output and the input as :
            `Tensor` of shape [... , output_dim, 1]
            `Tensor` of shape [... , output_dim, output_dim]
            `Tensor` of shape [... , output_dim, input_dim]
        """
        pass

    def loss_function(self, m_i, C_i, m_o, C_o=None, C_io=None):
        """
        Computes the loss function as required by GDSM.
        Must support batch processing.
        Note that the loss function is not aggregated since GDSM
        might need to ignore the losses associated to missing values.

        Args:
            m_i : Mean of the input as `Tensor` of shape
                [... , input_dim, 1]
            C_i : Covariance matrix of the input as `Tensor` of shape
                [... , input_dim, input_dim]
            m_o : Mean of the output as `Tensor` of shape
                [... , output_dim, 1]
            C_o : Covariance matrix of the output as `Tensor` of shape
                [... , output_dim, output_dim]
                Can be `None` to signify that the output is constant.
                (default None)
            C_io : Covariance matrix between input and output
                as `Tensor` of shape [... , input_dim, output_dim]
                Can be `None` to signify that the output is constant.
                (default None)

        Returns:
            A `Tensor` of shape [... , output_dim]
            containing the loss function at each output coordinate.
        """
        pass

    def sample(self, inputs):
        """
        Samples the function on given inputs (including noise).
        Must support batch processing.


        Args:
            inputs : Inputs as `Tensor` of shape [... , input_dim, 1]

        Returns:
            A `Tensor` of shape [... , output_dim, 1]
        """
        pass
