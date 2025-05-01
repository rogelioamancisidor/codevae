from tensorflow.keras import activations
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow as tf 
import numpy as np
from tensorflow.keras import mixed_precision

class ClsMNIST(layers.Layer):
    def __init__(self,
                 input_shape = (28,28,1),
                 filters     = 32,
                 kernel_size = 4,
                 strides     = 2,
                 latent_dim  = 20,
                 n_samples   = 1,
                 name        = 'clsMNIST',
                 **kwargs):
        super(ClsMNIST,self).__init__(name=name, **kwargs)
        self.hidden_layers = Sequential(
            [
            layers.InputLayer(input_shape=input_shape),
            layers.Conv2D(
                filters=filters,   kernel_size=kernel_size, strides=strides, padding='same'),
            layers.Activation(activations.relu),
            layers.Conv2D(
                filters=2*filters, kernel_size=kernel_size, strides=strides, padding='same'),
            layers.Activation(activations.relu),
            layers.Conv2D(
                filters=2*filters, kernel_size=kernel_size, strides=strides, padding='same'),
            layers.Activation(activations.relu),
            layers.Conv2D(
                filters=4*filters, kernel_size=kernel_size, strides=strides, padding='valid'),
            layers.Activation(activations.relu),
            layers.Dropout(0.5),
            ]
            )
        self.mu = layers.Dense(10)

    def call(self,inputs,training=True):
        x,y = inputs
        h = self.hidden_layers(x,training=training)
        mu = self.mu(h)
        out = tf.sigmoid(mu)
        out = tf.squeeze(out)
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.loss = bce(y,out)
        self.params = self.hidden_layers.trainable_variables + self.mu.trainable_variables

        return self.loss

    def test(self, inputs, training=False):
        x,y = inputs
        h = self.hidden_layers(x,training=training)
        mu = self.mu(h)
        mu = tf.squeeze(mu)
        pi_hat = tf.nn.softmax(mu)
        y_hat  = tf.one_hot(tf.math.argmax(pi_hat,1),10) 
        return y_hat
    
    def get_hidden(self,inputs,training=False):
        return self.hidden_layers(inputs,training=training)

    @tf.function
    def train(self, x, optimizer):
        with tf.GradientTape() as tape:
            loss = self.call(x)
        gradients = tape.gradient(self.loss, self.params)
        optimizer.apply_gradients(zip(gradients, self.params))

        return loss

class ClsText(layers.Layer):
    def __init__(self, dim=64, num_features=71):
        super(ClsText, self).__init__()
        self.conv1 = layers.Conv1D(filters=2*dim, kernel_size=1)
        self.resblock_1 = make_res_block_encoder(2*dim,3*dim,kernelsize=4,stride=2,padding='same',dilation=1)
        self.resblock_4 = make_res_block_encoder(3*dim,2*dim,kernelsize=4,stride=2,padding='valid',dilation=1)
        self.dropout = layers.Dropout(0.5)
        self.linear = layers.Dense(10) # 10 is the number of classes

    def call(self, inputs):
        x,y = inputs
        h = self.conv1(x)
        h = self.resblock_1(h)
        h = self.resblock_4(h)
        h = self.dropout(h)
        h = tf.squeeze(h)
        h = self.linear(h)
        out = tf.sigmoid(h)
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.loss = bce(y,out)
        self.params = self.conv1.trainable_variables + self.resblock_1.trainable_variables + self.resblock_4.trainable_variables + self.linear.trainable_variables

        return self.loss

    def test(self, inputs, training=False):
        x,y = inputs
        h = self.conv1(x)
        h = self.resblock_1(h)
        h = self.resblock_4(h)
        h = self.dropout(h)
        h = tf.squeeze(h)
        h = self.linear(h)
        pi_hat = tf.nn.softmax(h)
        y_hat  = tf.one_hot(tf.math.argmax(pi_hat,1),10) 
        return y_hat


    @tf.function
    def train(self, x, optimizer):
        with tf.GradientTape() as tape:
            loss = self.call(x)
        gradients = tape.gradient(self.loss, self.params)
        optimizer.apply_gradients(zip(gradients, self.params))

        return loss

def make_res_block_encoder(channels_in, channels_out, kernelsize, stride, padding, dilation):
    downsample = None
    if (stride != 1) or (channels_in != channels_out) or dilation != 1:
        downsample = Sequential([layers.Conv1D(filters=channels_out,
                                             kernel_size=kernelsize,
                                             strides=stride,
                                             padding=padding,
                                             dilation_rate=dilation),
                                 layers.BatchNormalization()])
    _layers = []
    _layers.append(ResidualBlockEncoder(channels_in, channels_out, kernelsize, stride, padding, dilation, downsample))
    return Sequential(*_layers)

class ResidualBlockEncoder(layers.Layer):
    def __init__(self,channels_in, channels_out, kernelsize, stride, padding, dilation, downsample, a=2, b=0.3):
        super(ResidualBlockEncoder, self).__init__()
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.conv1 = layers.Conv1D(filters=channels_in,  kernel_size=1, strides=1, padding='valid')
        self.conv2 = layers.Conv1D(filters=channels_out, kernel_size=kernelsize, strides=stride, padding=padding, dilation_rate=dilation)
        self.downsample = downsample
        self.relu = layers.Activation(activations.relu)
        self.dropout1 = layers.Dropout(0.5)
        self.dropout2 = layers.Dropout(0.5)
        self.a = a
        self.b = b

    def call(self, inputs, training=True):
        residual = inputs
        out = self.bn1(inputs)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.dropout1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.dropout2(out)
        if self.downsample:
            residual = self.downsample(inputs)
        out = self.a*residual + self.b*out
        return out

class ClsSVHN(layers.Layer):
    def __init__(self,
                 input_shape = (32,32,3),
                 filters     = 32,
                 kernel_size = 4,
                 strides     = 2,
                 latent_dim  = 20,
                 n_samples   = 1,
                 name        = 'clsSVHN',
                 **kwargs):
        super(ClsSVHN,self).__init__(name=name, **kwargs)
        self.hidden_layers = Sequential(
            [
            layers.InputLayer(input_shape=input_shape),
            layers.Conv2D(
                filters=filters,   kernel_size=kernel_size, strides=strides, padding='same'),
            layers.Dropout(0.5),
            layers.BatchNormalization(),
            layers.Activation(activations.relu),
            layers.Conv2D(
                filters=2*filters, kernel_size=kernel_size, strides=strides, padding='same'),
            layers.Dropout(0.5),
            layers.BatchNormalization(),
            layers.Activation(activations.relu),
            layers.Conv2D(
                filters=2*filters, kernel_size=kernel_size, strides=strides, padding='same'),
            layers.Dropout(0.5),
            layers.BatchNormalization(),
            layers.Activation(activations.relu),
            layers.Conv2D(
                filters=4*filters, kernel_size=kernel_size, strides=strides, padding='valid'),
            layers.Dropout(0.5),
            layers.BatchNormalization(),
            layers.Activation(activations.relu),
            ]
            )
        self.mu = layers.Dense(10)


    def call(self,inputs,training=True):
        x,y = inputs
        h = self.hidden_layers(x,training=training)
        mu = self.mu(h)
        out = tf.sigmoid(mu)
        out = tf.squeeze(out)
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.loss = bce(y,out)
        self.params = self.hidden_layers.trainable_variables + self.mu.trainable_variables

        return self.loss
    
    def test(self, inputs, training=False):
        x,y = inputs
        h = self.hidden_layers(x,training=training)
        mu = self.mu(h)
        mu = tf.squeeze(mu)
        pi_hat = tf.nn.softmax(mu)
        y_hat  = tf.one_hot(tf.math.argmax(pi_hat,1),10) 
        return y_hat

    def get_hidden(self,inputs,training=False):
        return self.hidden_layers(inputs,training=training)

    @tf.function
    def train(self, x, optimizer):
        with tf.GradientTape() as tape:
            loss = self.call(x)
        gradients = tape.gradient(self.loss, self.params)
        optimizer.apply_gradients(zip(gradients, self.params))

        return loss
