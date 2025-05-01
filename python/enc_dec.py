# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.models import Sequential
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
tfpl = tfp.layers
tfd  = tfp.distributions

class Weights(layers.Layer):
    def __init__(self, trainable=True, init_val=100., n_subsets=7,given_probs=False):
        super().__init__()
        self.u_w = tf.Variable(tf.repeat(init_val,repeats=n_subsets), trainable=trainable)
        self.given_probs = given_probs

    def call(self, inputs):
        if self.given_probs:
            w = self.u_w
        else:
            w = tf.nn.softmax(self.u_w)
        p =  tfd.Categorical(probs=w)
        return p

class EncMNIST_MLP(layers.Layer):
    def __init__(self,
                 input_shape = (784,),
                 units = 400,
                 activation  = 'relu',
                 latent_dim  = 20,
                 n_samples   = 1,
                 name        = 'encMNIST',
                 distribution = tfd.MultivariateNormalDiag,
                 **kwargs):
        super(EncMNIST_MLP,self).__init__(name=name, **kwargs)
        self.hidden_layers = Sequential(
            [
            layers.InputLayer(input_shape=input_shape),
            layers.Dense(units,activation=activation),
            layers.Dense(2*latent_dim),
            layers.Activation('linear', dtype='float32'),
            tfpl.DistributionLambda(lambda t: distribution(
                    t[..., :latent_dim], tf.math.exp(t[..., latent_dim:]))),
            ]
            )
        self._latent_dim = latent_dim

    @property
    def latent_dim(self):
        return self._latent_dim

    def call(self, inputs):
        inputs = tf.reshape(inputs,[-1,inputs.shape[1]*inputs.shape[2]])
        qz_x = self.hidden_layers(inputs)

        return qz_x

class DecMNIST_MLP(layers.Layer):
    def __init__(self,
                 latent_dim = 20,
                 units = 400,
                 activation  = 'relu',
                 output_dim  = 784,
                 target_shape=(28,28,1),
                 n_samples   = 1,
                 name        = 'decMNIST',
                 distribution = tfd.Laplace,
                 **kwargs):
        super(DecMNIST_MLP,self).__init__(name=name, **kwargs)
        self.hidden_layers = Sequential(
            [
            layers.InputLayer(input_shape=latent_dim),
            layers.Dense(units,activation=activation),
            layers.Dense(output_dim),
            layers.Activation('linear', dtype='float32'),
            ]
            )
        self.distribution=distribution
        if self.distribution.__name__ == 'Bernoulli':
            self.s2 = Sequential([layers.InputLayer(input_shape=(784,)),layers.Flatten(),tfpl.IndependentBernoulli(target_shape)])

    def call(self, inputs):
        out = self.hidden_layers(inputs)
        if self.distribution.__name__ == 'Laplace':
            out = tf.reshape(out, (-1,28,28,1))
            #out = tf.math.sigmoid(out)
            px_z = self.distribution(out, 0.75)
        elif self.distribution.__name__ == 'Bernoulli':
            px_z = self.s2(out)

        return px_z

class EncSVHN(layers.Layer):
    def __init__(self,
                 input_shape = (32,32,3),
                 filters     = 32,
                 kernel_size = 4,
                 strides     = 2,
                 activation  = 'relu',
                 latent_dim  = 20,
                 n_samples   = 1,
                 distribution = tfd.MultivariateNormalDiag,
                 name        = 'encSVHN',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.hidden_layers = Sequential(
            [
            layers.InputLayer(input_shape=input_shape),
            layers.Conv2D(
                filters=filters,   kernel_size=kernel_size, strides=strides, activation=activation, padding='same'),
            layers.Conv2D(
                filters=2*filters, kernel_size=kernel_size, strides=strides, activation=activation, padding='same'),
            layers.Conv2D(
                filters=2*filters, kernel_size=kernel_size, strides=strides, activation=activation, padding='same'),
            layers.Conv2D(
                filters=4*filters, kernel_size=kernel_size, strides=strides, activation=activation, padding='valid'),
            layers.Flatten(),
            layers.Dense(2*latent_dim),
            layers.Activation('linear', dtype='float32'),
            tfpl.DistributionLambda(lambda t: distribution(
                    t[..., :latent_dim], tf.math.exp(t[..., latent_dim:]))),
            ]
            )
        self._latent_dim = latent_dim

    @property
    def latent_dim(self):
        return self._latent_dim

    def call(self, inputs):
        qz_x = self.hidden_layers(inputs)

        return qz_x

class DecSVHN(layers.Layer):
    def __init__(self,
                 latent_dim=20,
                 target_shape=(32,32,3),
                 filters=32,
                 kernel_size=4,
                 activation='relu',
                 name = 'decoderCNN',
                 distribution='tfd.Laplace',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        units = np.prod(target_shape) 
        self.hidden_layers = Sequential(
                [
                layers.InputLayer(input_shape=(latent_dim,)),
                layers.Dense(128),
                layers.Reshape([1, 1, 128]),
                layers.Conv2DTranspose(
                    filters=filters*2, kernel_size=kernel_size, strides=1, padding='valid',
                    activation=activation),
                layers.Conv2DTranspose(
                    filters=filters*2, kernel_size=kernel_size, strides=2, padding='same',
                    activation=activation),
                layers.Conv2DTranspose(
                    filters=filters, kernel_size=kernel_size, strides=2, padding='same',
                    activation=activation),
                layers.Conv2DTranspose(
                    filters=3, kernel_size=4, strides=2, padding='same'),
                layers.Activation('linear', dtype='float32'),
                ]
                )
        self.distribution=distribution
        if self.distribution.__name__ == 'Bernoulli':
            self.s2 = Sequential([layers.InputLayer(input_shape=(32,32,3)),layers.Flatten(),tfpl.IndependentBernoulli(target_shape)])
   
    def call(self, inputs):
        out = self.hidden_layers(inputs)
        if self.distribution.__name__ == 'Laplace':
            px_z = self.distribution(out, 0.75)
        elif self.distribution.__name__ == 'Bernoulli':
            px_z = self.s2(out)

        return px_z

class EncText(layers.Layer):
    # class dim is the dim latent space
    def __init__(self, num_characters=8, dim=64, num_features=71, latent_dim=20, n_samples=1, distribution = tfd.MultivariateNormalDiag):
        super(EncText, self).__init__()
        self.hidden_layers = Sequential(
                            [
                            layers.InputLayer(input_shape=(num_characters,num_features)),
                            FeatureEncText(dim, num_features),
                            layers.Dense(2 * latent_dim),
                            layers.Activation('linear', dtype='float32'),
                            tfpl.DistributionLambda(lambda t: distribution(
                                    t[..., :latent_dim], tf.math.exp(t[..., latent_dim:]))),
                            ]
                            )    
        
        self._latent_dim = latent_dim
        
    @property
    def latent_dim(self):
        return self._latent_dim

    def call(self, inputs):
        qz_x = self.hidden_layers(inputs)

        return qz_x

class FeatureEncText(layers.Layer):
    def __init__(self, dim=64, num_features=71):
        super(FeatureEncText, self).__init__()
        self.dim = dim
        self.conv1 = layers.Conv1D(filters=2*self.dim, kernel_size=1, activation='relu')
        self.conv2 = layers.Conv1D(filters=2*self.dim, kernel_size=4, strides=2, padding='same', activation='relu')
        self.conv3 = layers.Conv1D(filters=2*self.dim, kernel_size=4, strides=2, padding='valid', activation='relu')

    def call(self,x):
        out = self.conv1(x);
        out = self.conv2(out);
        out = self.conv3(out);
        h = tf.reshape(out, [-1,2*self.dim])
        
        return h

class DecText(layers.Layer):
    def __init__(self, 
                dim=64, 
                latent_dim=20,
                num_features=71, 
                num_characters=8,
                distribution=tfpl.OneHotCategorical
                ):
        super(DecText, self).__init__()
        self.hidden_layers = Sequential([
                layers.InputLayer(input_shape=latent_dim),
                layers.Dense(2*dim),
                layers.Reshape(target_shape=(1,2*dim)),
                layers.Conv1DTranspose(2*dim, kernel_size=4, strides=1, padding='valid', activation='relu'),
                layers.Conv1DTranspose(2*dim, kernel_size=4, strides=2, padding='same', activation='relu'),
                layers.Conv1D(num_features, kernel_size=1),
                layers.Dense(tfpl.OneHotCategorical.params_size(num_features)),
                layers.Activation('linear', dtype='float32'),
                distribution(num_features)
                ])
    
    def call(self, inputs):
        pz_x = self.hidden_layers(inputs)
        
        return pz_x
