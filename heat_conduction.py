import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import pinns as pn

initializer = keras.initializers.RandomUniform(minval=-10, maxval=10)
NN = keras.Sequential([
    keras.layers.Dense(10, input_shape = (2,)),
    keras.layers.Dense(100, activation = 'tanh'),
    keras.layers.Dense(100, activation = 'tanh'),
    keras.layers.Dense(100, activation = 'tanh'),
    keras.layers.Dense(1, activation = None)
])

lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries = [6000, 9000], values = [0.1, 0.01, 0.001])

optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
#optimizer = tf.keras.optimizers.Adam(learning_rate = 0.1)
NN.compile(
    optimizer = optimizer,
    loss = tf.keras.losses.MeanSquaredError()
)
#number of samples to draw for each step, for boundary, initial and pde
batch_size = 100

#simulation parameters
t_max = 1#1200.0
x_max = 1#3.81*0.01
T_0 = 300
T_L = 310
T_R = 300

c = 381.0
k = 32.0
rho = 6093
pinn = pn.pinn_1d(NN, t_max, x_max, T_L, T_R)
#input for plotting
test_input = tf.transpose(tf.concat([[tf.ones(50)*t_max/2/t_max], [tf.linspace(0.0, x_max, 50)/x_max]], axis=0))

#static input for verifcation for all 3 losses:
x_test = pinn.sample_x(batch_size)
t_test = pinn.sample_t(batch_size)
verify_input_R = pinn.sample_R(batch_size)
verify_input_L = pinn.sample_L(batch_size)
verify_input_initial = pinn.sample_initial(batch_size)

X_test, T_test = tf.meshgrid(x_test, t_test)

#training loop
for i in range(10001):

    #creating training input:
    x_input = pinn.sample_x(batch_size)
    t_input = pinn.sample_t(batch_size)
    initial_input = pinn.sample_initial(batch_size)
    input_R = pinn.sample_R(batch_size)
    input_L = pinn.sample_L(batch_size)

    with tf.GradientTape() as training_tape:
        #outer gradient tape for traininig
        training_tape.watch([pinn.variables])
        
        with tf.GradientTape() as tape1:
            #inner gradient tape to calculate physical loss
            tape1.watch([t_input, x_input])

            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch([t_input, x_input])
                #guesses temperature
                T = pinn.cast_tx(t_input, x_input)

        #derivatives
            dt = tf.cast(tape2.gradient(T, t_input), 'float32')
            dx = tape2.gradient(T, x_input)

        dxx = tape1.gradient(dx, x_input)

        dx = tf.cast(dx, 'float32')
        dxx = tf.cast(dxx, 'float32')

        #loss calculation
        f = rho*c*dt - k * dxx
        physical_loss = tf.reduce_mean(f**2)

        T_0_guess = pinn.cast(initial_input)
        initial_loss = tf.reduce_mean((T_0_guess - T_0)**2)

        T_L_guess = pinn.cast(input_L)
        T_R_guess = pinn.cast(input_R)
        boundary_loss = tf.reduce_mean((T_L_guess-T_L)**2) + tf.reduce_mean((T_R_guess-T_R)**2)

        loss = physical_loss + boundary_loss + initial_loss # boundary_loss +

    #verify:    
    with tf.GradientTape(persistent=True) as tape:

        #inner gradient tape to calculate physical loss
        tape.watch([X_test, T_test])

        #guesses temperature
        T_verify = pinn.cast_tx(T_test, X_test)

    #derivatives
    dt_test = tf.cast(tape.gradient(T_verify, T_test), 'float32')
    dx_test = tape.gradient(T_verify, X_test)
    dxx_test = tape.gradient(dx_test, X_test)

    dx_test = tf.cast(dx_test, 'float32')
    dxx_test = tf.cast(dx_test, 'float32')

    #loss calculation
    f_test = rho*c*dt_test - k * dxx_test
    physical_loss_test = tf.reduce_mean(f_test**2)

    T_0_guess_test = pinn.cast(verify_input_initial)
    initial_loss_test = tf.reduce_mean((T_0_guess_test - T_0)**2)

    T_L_guess_test = pinn.cast(verify_input_L)
    T_R_guess_test = pinn.cast(verify_input_R)
    boundary_loss_test = tf.reduce_mean((T_L_guess_test-T_L)**2) + tf.reduce_mean((T_R_guess_test-T_R)**2)

    loss_test =  [boundary_loss_test, physical_loss_test,  initial_loss_test]

    if i % 100 == 0:
        print(loss_test)

    #training
    grad = training_tape.gradient(loss, pinn.variables)
    optimizer.apply_gradients(zip(grad, pinn.variables))

    if i % 1000 == 0:
        plt.plot(tf.linspace(0.0, x_max, 50), pinn.cast(test_input), label = i)

pinn.save('pinn')
plt.legend()
plt.show()
