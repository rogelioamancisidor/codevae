import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from CoDEVAE import CoDEVAE
from Modality import Modality
import tensorflow as tf
import tensorflow_probability as tfp
tfpl = tfp.layers
tfd = tfp.distributions
from tensorflow.keras import mixed_precision
import numpy as np

class CoDEVAE_MMNIST(CoDEVAE, tf.keras.Model):
    def __init__(self, 
                 latent_dim, 
                 latent_dim_s, 
                 mnist_enc, mnist_dec,
                 correlation=0.5,
                 correlation_matrix=None,
                 prior=tfd.MultivariateNormalDiag, 
                 distribution_enc=tfd.MultivariateNormalDiag,
                 distribution_dec=tfd.Laplace,
                 code_prior='flat', 
                 fusion_method='code',
                 subsets_trainable_weights = True,
                 beta = 2.5,
                 num_modalities=2,
                 train=True,
                 modality_specific=False,
                 given_weights=True,
                 **kwargs):
        CoDEVAE.__init__(self,latent_dim, prior=prior, fusion_method=fusion_method, distribution_fusion=distribution_enc, code_prior=code_prior, correlation=correlation, correlation_matrix=correlation_matrix, num_modalities=num_modalities)
        tf.keras.Model.__init__(self,**kwargs)
        self.latent_dim = latent_dim
        self.latent_dim_s = latent_dim_s
        self.distribution_enc = distribution_enc
        self.distribution_dec = distribution_dec
        self.modality_specific = modality_specific
        
        # initialize abstract method
        self.modalities(num_modalities, mnist_enc, mnist_dec)
        # call get_subsets
        self.get_subsets()
        # call rec_weights, pass modality with more no of params. Doesnt matter in this case
        self.rec_weights('m0')

        # beta to scale KL
        self.beta = beta
        # common beta style for all modalities
        factor=1

        # weights network
        self.subsets_trainable_weights = subsets_trainable_weights
        if given_weights:
            self.weights_networks(trainable=False, n_subsets=1, init_val=weights, given_probs=True)
            self.beta_style = {'m0':1*factor,'m1':1*factor,'m2':1*factor,'m3':1*factor,'m4':1*factor}
        else:
            self.weights_networks(trainable=subsets_trainable_weights, n_subsets=len(self.subsets.keys()))
            self.beta_style = {'m0':1*factor,'m1':3*factor,'m2':4*factor,'m3':5*factor,'m4':2*factor}

        # get model params 
        self.set_model_params()

        if modality_specific:
            print('model with modality specific latent spaces')
            self.prior_style  = tfd.MultivariateNormalDiag(tf.zeros(latent_dim_s), tf.ones(latent_dim_s))

        if train:
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            print('Compute dtype: %s' % policy.compute_dtype)
            print('Variable dtype: %s' % policy.variable_dtype)
        print('MMNIST model with subsets',self.subsets.keys())

    # overriding so we can use a different dim for style 
    # XXX Maybe this should be coded in  superclass 
    def style_kl(self, inputs):
        kl_loss = dict()

        for m in self.modalities.keys():
            x = inputs[m] 
            style_expert = self.modalities[m]
            qs_x = style_expert.qz_x(x,modality_specific=True)
            
            # I simply take KL outside if  
            kl   = tfd.kl_divergence(qs_x, self.prior_style)
            kl_loss[m] = kl
        
        return kl_loss
    
    def set_model_params(self):
        self.params = self.weights_nets_params + self.modalities_params
        trainableParams = np.sum([np.prod(v.get_shape()) for v in self.weights_nets_params])

    def modalities(self, num_modalities, mnist_enc, mnist_dec):
        modalities = dict()
        modalities_params = [] 
        for i in range(num_modalities):
            name = 'm'+str(i)
            modalities[name] = Modality(name, mnist_enc, mnist_dec, self.distribution_enc, self.distribution_dec, (28,28,3),latent_dim=self.latent_dim, modality_specific=self.modality_specific,latent_dim_s=self.latent_dim_s)
            modalities_params+=modalities[name].params
            
        self.modalities = modalities
        self.modalities_params = modalities_params

    def call(self, inputs):
        # both losses are dictionaries with 2^M-1 subsets
        kl_loss    = self.subsets_kl(inputs)
        recon_loss = self.subsets_reconstruction(inputs,modality_specific=self.modality_specific)
        weights = self.subsets_weights()
        if self.modality_specific:
            kl_style = self.style_kl(inputs)
        
        codevae_loss = 0
        losses = dict()
        
        if self.subsets_trainable_weights:
            H = weights['H']
        else:
            H = 0
        
        for k in self.subsets.keys():
            # here I just save some 'losses' to track
            losses[k] = {'kl':tf.reduce_mean(kl_loss[k]), 'recon':tf.reduce_mean(recon_loss[k]), 'weight':tf.reduce_mean(weights[k])}
            
            # k'th elbo
            k_elbo = weights[k]*(tf.reduce_mean(recon_loss[k]) - tf.reduce_mean(self.beta*kl_loss[k])) 
            codevae_loss+=-k_elbo
        
        style_kl = 0
        if self.modality_specific:
            for m in self.modalities.keys():
                # XXX I use + KL as at the end we minimize -ELBO
                style_kl += weights[m]*(tf.reduce_mean(self.beta_style[m]*kl_style[m]))
        
        self.codevae_loss = codevae_loss - 1000*H + style_kl 
        losses['codevae_loss'] = codevae_loss
        return losses

    @tf.function
    def train(self, inputs, optimizer):
        with tf.GradientTape() as tape:
            losses = self.call(inputs)
            scaled_loss = optimizer.get_scaled_loss(self.codevae_loss)
        scaled_gradients = tape.gradient(scaled_loss, self.params)
        gradients = optimizer.get_unscaled_gradients(scaled_gradients)
        optimizer.apply_gradients(zip(gradients, self.params))

        return losses

    @tf.function
    def evaluate(self, inputs, calculate_llik=False, batch_size=None, modality_specific=False, z_method='sample'):
        if calculate_llik:
            print('calculating log-likelihood...')
            # llik
            return self.loglikelihood(inputs,batch_size,modality_specific=modality_specific)
        else:
            print('generating z reps...')
            all_z = self.latent_representations(inputs, training=False)
            print('generating modalities...')
            x_hat = self.generate_all_modalities(inputs, training=False, modality_specific=modality_specific, z_method=z_method)

            return all_z, x_hat
