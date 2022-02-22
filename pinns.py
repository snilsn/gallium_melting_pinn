import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt

pinn = keras.Sequential([
    keras.layers.Dense(100, input_shape = (1,)),
    keras.layers.Dense(100, activation = 'tanh'),
    keras.layers.Dense(100, activation = 'tanh'),
    keras.layers.Dense(1, activation=None)
])

pinn.compile(
    optimizer = tf.keras.optimizers.Adam(), 
    loss = tf.keras.losses.MeanSquaredError()
)

t = tf.linspace(0, 20, 100)
t_input = tf.transpose(tf.concat([[t]], axis = 0))

M_0 = 1.0
l = 1.0
t_0 = tf.constant([[0.0]])

def analytical(M_0, l, t):
    return M_0*tf.exp(-l*t)

optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)

start = pinn(t_input)
for i in range(100):

    with tf.GradientTape() as tape2:
        tape2.watch([pinn.trainable_variables, t_0])
        with tf.GradientTape() as tape1:
            tape1.watch([t_input])
            m = pinn(t_input)
    
        dt = tf.cast(tape1.gradient(m, t_input), 'float32')

        f = dt + l*m
        physical_loss = tf.reduce_mean(f)

        m0 = pinn(t_0)
        initial_loss = (m0-M_0)**2

    loss = physical_loss + initial_loss
    grad = tape2.gradient(loss, pinn.trainable_variables)

    optimizer.apply_gradients(zip(grad, pinn.trainable_variables))

    if i % 10 == 0:
        plt.plot(t_input, pinn(t_input), label = i+1)

plt.plot(t_input, start, label = 'initial guess')
plt.plot(t_input, analytical(M_0, l, t_input), label = 'analytical solution')
plt.legend()
plt.grid()
plt.show()