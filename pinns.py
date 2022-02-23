import tensorflow as tf
import tensorflow.keras as keras

class pinn:

    def __init__(self, NN, t_0, t, M_0, l):

        self.pinn = NN
        self.variables = NN.trainable_variables
        self.optimizer = NN.optimizer
        self.t_0 = t_0
        self.t = t
        self.t_input = tf.transpose(tf.concat([[self.t]], axis = 0))
        self.M_0 = M_0
        self.l = l

    def cast(self):
        return self.pinn(self.t_input)

    def train(self):

        with tf.GradientTape() as tape2:

            tape2.watch([self.variables, self.t_0])

            with tf.GradientTape() as tape1:

                tape1.watch([self.t_input])
                m = self.cast()
    
            dt = tf.cast(tape1.gradient(m, self.t_input), 'float32')

            self.f = dt + self.l*m
            self.physical_loss = tf.reduce_mean(self.f**2)

            m0 = self.pinn(self.t_0)
            self.initial_loss = (m0-self.M_0)**2

            self.loss = self.physical_loss + self.initial_loss

        grad = tape2.gradient(self.loss, self.variables)
        self.optimizer.apply_gradients(zip(grad, self.variables))