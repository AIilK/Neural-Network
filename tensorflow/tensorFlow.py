import numpy as np
import tensorflow as tf

w = tf.Variable(0.0, dtype=tf.float32)
train = tf.keras.optimizers.SGD(learning_rate=0.01)
x = tf.placeholder

def cost():
    return w**2 - 10 * w + 25

for step in range(1000):  
    with tf.GradientTape() as tape:
        loss = cost()
    grad = tape.gradient(loss, [w])
    train.apply_gradients(zip(grad, [w]))


print(w.numpy())
