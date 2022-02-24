import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt

initializer = keras.initializers.RandomUniform(minval=-1, maxval=1)
pinn = keras.Sequential([
    keras.layers.Dense(50, input_shape = (2,), kernel_initializer=initializer),
    keras.layers.Dense(50, activation = 'tanh', kernel_initializer=initializer),
    keras.layers.Dense(50, activation = 'tanh', kernel_initializer=initializer),
    keras.layers.Dense(50, activation = 'tanh', kernel_initializer=initializer),
    keras.layers.Dense(1, activation = 'tanh', kernel_initializer=initializer)
])

lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries = [3000, 10000, 18000], values = [0.1, 0.01, 0.001, 0.0001])

#optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
pinn.compile(
    optimizer = optimizer,
    loss = tf.keras.losses.MeanSquaredError()
)

#number of samples to draw for each step, for boundary, initial and pde
batch_size = 200

#simulation parameters
t_min = 0.0
t_max = 1200.0
x_min = 0.0
x_max = 3.81*0.01
T_0 = 300
T_L = 1
T_R = 1

c = 381.0
k = 32.0
rho = 6093

#input for plotting
test_input = tf.transpose(tf.concat([[tf.ones(50)*t_max/2], [tf.linspace(x_min, x_max, 50)]], axis=0))

#static input for verifcation for all 3 losses:
x_test = tf.linspace(x_min, x_max, 20)
t_test = tf.linspace(t_min, t_max, 20)

X_test, T_test = tf.meshgrid(x_test, t_test)

x_test_boundary_R = tf.ones(20)*x_max
t_R_input = tf.linspace(t_min, t_max, 20)
verify_input_R = tf.transpose(tf.concat([[t_R_input], [x_test_boundary_R]], axis=0))

x_test_boundary_L = tf.ones(20)*x_min
t_L_input = tf.linspace(t_min, t_max, 20)
verify_input_L = tf.transpose(tf.concat([[t_L_input], [x_test_boundary_L]], axis=0))

t_initial_test = tf.zeros(20)
verify_input_initial = tf.transpose(tf.concat([[t_initial_test], [x_test]], axis=0))

#training loop
for i in range(1001):

    #creating training input:
    x_input = tf.random.uniform(
        [batch_size], minval=x_min, maxval=x_max, dtype=tf.dtypes.float32
    )
    t_input = tf.random.uniform(
        [batch_size], minval=t_min, maxval=t_max, dtype=tf.dtypes.float32
    )

    x_initial_input = tf.random.uniform(
        [batch_size], minval=x_min, maxval=x_max, dtype=tf.dtypes.float32
    )
    t_initial_input = tf.zeros(batch_size)

    x_L_input = tf.ones(batch_size)*x_min
    t_L_input = tf.random.uniform(
        [batch_size], minval=t_min, maxval=t_max, dtype=tf.dtypes.float32
    )

    x_R_input = tf.ones(batch_size)*x_max
    t_R_input = tf.random.uniform(
        [batch_size], minval=t_min, maxval=t_max, dtype=tf.dtypes.float32
    )

    with tf.GradientTape() as training_tape:
        #outer gradient tape for traininig
        training_tape.watch([pinn.trainable_variables])
        
        with tf.GradientTape() as tape1:
            #inner gradient tape to calculate physical loss
            tape1.watch([t_input, x_input])
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch([t_input, x_input])
                input_values = tf.transpose(tf.concat([[t_input], [x_input]], axis = 0))
                #guesses temperature
                T = pinn(input_values)

        #derivatives
            dt = tf.cast(tape2.gradient(T, t_input), 'float32')
            dx = tape2.gradient(T, x_input)
        dxx = tape1.gradient(dx, x_input)

        dx = tf.cast(dx, 'float32')
        dxx = tf.cast(dx, 'float32')

        #loss calculation
        #f = rho*c*dt - k * dxx
        f = dxx - 1
        physical_loss = tf.reduce_mean(f**2)

        initial_input_values = tf.transpose(tf.concat([[t_initial_input], [x_initial_input]], axis = 0))
        T_0_guess = pinn(initial_input_values)
        initial_loss = tf.reduce_mean((T_0_guess - T_0)**2)

        T_L_guess = pinn(tf.transpose(tf.concat([[t_L_input], [x_L_input]], axis = 0)))
        T_R_guess = pinn(tf.transpose(tf.concat([[t_R_input], [x_R_input]], axis = 0)))
        boundary_loss = tf.reduce_mean((T_L_guess-T_L)**2) + tf.reduce_mean((T_R_guess-T_R)**2)

        loss =  physical_loss#+ initial_loss # boundary_loss +

    #verify:    
    with tf.GradientTape(persistent=True) as tape:

        #inner gradient tape to calculate physical loss
        tape.watch([X_test, T_test])

        #guesses temperature
        verify_input = tf.transpose(tf.concat([[T_test], [X_test]], axis=0))
        T_verify = pinn(verify_input)

    #derivatives
    dt_test = tf.cast(tape.gradient(T_verify, T_test), 'float32')
    dx_test = tape.gradient(T_verify, X_test)
    dxx_test = tape.gradient(dx_test, X_test)

    dx_test = tf.cast(dx_test, 'float32')
    dxx_test = tf.cast(dx_test, 'float32')

    #loss calculation
    #f_test = rho*c*dt_test - k * dxx_test
    f_test = dxx +1
    physical_loss_test = tf.reduce_mean(f_test**2)

    T_0_guess_test = pinn(verify_input_initial)
    initial_loss_test = tf.reduce_mean((T_0_guess_test - T_0)**2)

    T_L_guess_test = pinn(verify_input_L)
    T_R_guess_test = pinn(verify_input_R)
    boundary_loss_test = tf.reduce_mean((T_L_guess_test-T_L)**2) + tf.reduce_mean((T_R_guess_test-T_R)**2)

    loss_test =  [boundary_loss_test, physical_loss_test,  initial_loss_test]

    if i % 100 == 0:
        print(loss_test)

    #training
    grad = training_tape.gradient(loss, pinn.trainable_variables)
    optimizer.apply_gradients(zip(grad, pinn.trainable_variables))

    if i % 1000 == 0:
        plt.plot(tf.linspace(x_min, x_max, 50), pinn(test_input), label = i)

pinn.save('pinn')
plt.legend()
plt.show()
