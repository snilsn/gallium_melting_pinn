import tensorflow as tf 
import tensorflow.keras as keras
import matplotlib.pyplot as plt

pinn = keras.models.load_model('pinn')

t_min = 0.0
t_max = 1
x_min = 0.0
x_max = 1
#test_input = tf.transpose(tf.concat([[tf.ones(50)*t_max/10*10], [tf.linspace(x_min, x_max, 50)]], axis=0))
#plt.plot(tf.linspace(x_min, x_max, 50), pinn(test_input))
#plt.show()

for i in range(11):
    test_input = tf.transpose(tf.concat([[tf.ones(50)*t_max/10*i/t_max], [tf.linspace(x_min, x_max, 50)/t_max]], axis=0))
    plt.plot(tf.linspace(x_min, x_max, 50), pinn(test_input), label = i*t_max/10)

plt.legend()
plt.show()