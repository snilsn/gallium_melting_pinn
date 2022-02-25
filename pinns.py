import tensorflow as tf
import tensorflow.keras as keras

class pinn_1d:

    def __init__(self, NN, t_max, x_max, y_L , y_R):

        self.pinn = NN
        self.variables = NN.trainable_variables
        self.optimizer = NN.optimizer
        self.t_max = t_max
        self.x_max = x_max
        self.y_R = y_R
        self.y_L = y_L

    def uniform_x(self, batch_size):
        x_data = tf.linspace(0.0,self.x_max, 50)
        return x_data

    def sample_x(self, batch_size):

        x_data = tf.random.uniform(
        [batch_size], minval=0.0, maxval=self.x_max, dtype=tf.dtypes.float32
        )
        return x_data

    def sample_t(self, batch_size):

        t_data = tf.random.uniform(
        [batch_size], minval=0.0, maxval=self.t_max, dtype=tf.dtypes.float32
        )
        return t_data

    def sample_initial(self, batch_size):

        t_data = tf.zeros(batch_size)
        x_data = tf.random.uniform(
        [batch_size], minval=0.0, maxval=self.x_max, dtype=tf.dtypes.float32
        )

        initial_input = tf.transpose(tf.concat([[t_data], [x_data]], axis=0))

        return initial_input

    def sample_R(self, batch_size):

        t_data = tf.random.uniform(
        [batch_size], minval=0.0, maxval=self.t_max, dtype=tf.dtypes.float32
        )
        x_data = tf.ones(batch_size)*self.x_max

        input_R = tf.transpose(tf.concat([[t_data], [x_data]], axis=0))

        return input_R

    def sample_L(self, batch_size):

        t_data = tf.random.uniform(
        [batch_size], minval=0.0, maxval=self.t_max, dtype=tf.dtypes.float32
        )
        x_data = tf.zeros(batch_size)

        input_L = tf.transpose(tf.concat([[t_data], [x_data]], axis=0))

        return input_L

    def sample_input(self, batch_size):
        x = sample_x(batch_size)
        t = sample_t(batch_size)
        pinn_input = tf.transpose(tf.concat([[t], [x]], axis=0))
        return pinn_input

    def cast(self, input_data):
        output = self.pinn(input_data)
        return output

    def cast_tx(self, t, x):
            pinn_input = tf.transpose(tf.concat([[t], [x]], axis=0))
            output = self.cast(pinn_input)
            return output

    def save(self, name):
        self.pinn.save(name)