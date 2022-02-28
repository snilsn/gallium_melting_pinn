import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
plt.rc('font', size = 14)
import pinns as pn

def analytical(M_0, l, t):
    return M_0*tf.exp(-l*t)

optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)
t_max = 5
t_test = tf.transpose(tf.concat([[tf.linspace(0, t_max, 50)]], axis=0))
M_0 = 50.0
l = 1.0
batch_size = 10

for acti in [None, 'swish', 'tanh', 'softplus', 'sigmoid','softmax', 'selu', 'relu']:

    NN = keras.Sequential([
        keras.layers.Dense(100, input_shape = (1,), activation = acti),
        keras.layers.Dense(100, activation = acti),
        keras.layers.Dense(100, activation = acti),
        keras.layers.Dense(100, activation = acti),
        keras.layers.Dense(1, activation = None)
    ])
    NN.compile(
        optimizer = optimizer, 
        loss = tf.keras.losses.MeanSquaredError()
    )
    pinn = pn.pinn_0d(NN, t_max, 0.0)

    for i in range(1000):
        
        t = pinn.sample_input(batch_size)

        with tf.GradientTape() as train_tape:

            train_tape.watch(pinn.variables)

            with tf.GradientTape() as tape:
                tape.watch(t)
                M = pinn.cast(t)
            dt = tape.gradient(M, t)
            dt = tf.cast(dt, 'float32')

            f = dt + l*M

            physical_loss = tf.reduce_mean(f**2)

            M_0_guess = pinn.cast(pinn.sample_initial(batch_size))
            initial_loss = tf.reduce_mean((M_0_guess - M_0)**2)
            loss = physical_loss + initial_loss

        grad = train_tape.gradient(loss, pinn.variables)
        optimizer.apply_gradients(zip(grad, pinn.variables))

    plt.plot(t_test, pinn.cast(t_test), label = acti)

plt.plot(t_test, analytical(M_0, l, t_test), 'kx', label = 'analytical solution')

plt.xlabel('t[s]')
plt.ylabel('m[arbitrary units]')
plt.legend()
plt.grid()
plt.show()