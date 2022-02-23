import matplotlib.pyplot as plt
import tensorflow.keras as keras
import tensorflow as tf
import pinns

def analytical(M_0, l, t):
    return M_0*tf.exp(-l*t)

NN = keras.Sequential([
    keras.layers.Dense(10, input_shape = (1,)),
    keras.layers.Dense(10, activation = 'tanh'),
    keras.layers.Dense(10, activation = 'tanh'),
    keras.layers.Dense(1, activation=None)
])

NN.compile(
    optimizer = tf.keras.optimizers.Adam(), 
    loss = tf.keras.losses.MeanSquaredError()
)

t = tf.linspace(0, 5, 20)
t_input = tf.transpose(tf.concat([[t]], axis = 0))

M_0 = 1.0
l = 1.0
t_0 = tf.constant([[0.0]])

pinn = pinns.pinn(NN, t_0, t, M_0, l)

start = pinn.cast()

for i in range(100):
    pinn.train()

plt.plot(t_input, pinn.cast(), label = 'after training')
plt.plot(t_input, start, label = 'initial guess')
plt.plot(t_input, analytical(M_0, l, t_input), label = 'analytical solution')
plt.legend()
plt.grid()
plt.show()