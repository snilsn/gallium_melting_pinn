import tensorflow as tf 
import tensorflow.keras as keras
import matplotlib.pyplot as plt

pinn = keras.models.load_model('pinn')

t_min = 0.0
t_max = 1200.0
x_min = 0.0
x_max = 3.81*0.01
T_0 = 300.0
T_L = 310.0
T_R = 0.0

#test_input = tf.transpose(tf.concat([[tf.ones(50)*t_max/10*10], [tf.linspace(x_min, x_max, 50)]], axis=0))
#plt.plot(tf.linspace(x_min, x_max, 50), pinn(test_input))
#plt.show()

for i in range(11):
    test_input = tf.transpose(tf.concat([[tf.ones(50)*t_max/10*i], [tf.linspace(x_min, x_max, 50)]], axis=0))
    plt.plot(tf.linspace(x_min, x_max, 50), pinn(test_input), label = i)

plt.legend()
plt.show()