import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from CoDEVAE import CoDEVAE
from Modality import Modality
import tensorflow as tf
import tensorflow_probability as tfp
tfpl = tfp.layers
tfd = tfp.distributions
from tensorflow.keras import mixed_precision

class CoDEVAE_MST(CoDEVAE, tf.keras.Model):
    def __init__(self, 
                 latent_dim, 
                 mnist_enc, mnist_dec,
                 svhn_enc, svhn_dec,
                 text_enc, text_dec,
                 correlation=0.5,
                 correlation_matrix=None,
                 prior=tfd.MultivariateNormalDiag, 
                 distribution_enc=tfd.MultivariateNormalDiag,
                 distribution_dec=tfd.Laplace,
                 code_prior='flat', 
                 fusion_method='code',
                 subsets_trainable_weights = True,
                 rec_weights=None,
                 beta = 5.0,
                 train=True,
                 **kwargs):
        CoDEVAE.__init__(self,latent_dim, prior=prior, fusion_method=fusion_method, distribution_fusion=distribution_enc, code_prior=code_prior, correlation=correlation, correlation_matrix=correlation_matrix)
        tf.keras.Model.__init__(self,**kwargs)
        self.latent_dim = latent_dim
        self.distribution_enc = distribution_enc
        self.distribution_dec = distribution_dec
        
        # initialize abstract method
        self.modalities(mnist_enc, mnist_dec, svhn_enc, svhn_dec, text_enc, text_dec)

        # call get_subsets
        self.get_subsets()
        # call rec_weights, pass modality with more no of params
        self.rec_weights('svhn',rec_weights=rec_weights)
        # beta to scale KL
        self.beta = beta
        # weights network
        self.subsets_trainable_weights = subsets_trainable_weights
        self.weights_networks(trainable=subsets_trainable_weights, n_subsets=len(self.subsets.keys()))
        # get model params 
        self.set_model_params()
        
        if train:
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            print('Compute dtype: %s' % policy.compute_dtype)
            print('Variable dtype: %s' % policy.variable_dtype)

    def set_model_params(self):
        self.params = self.weights_nets_params + self.modalities_params

    def modalities(self, mnist_enc, mnist_dec, svhn_enc, svhn_dec, text_enc, text_dec):
        MNIST = Modality('mnist', mnist_enc, mnist_dec, self.distribution_enc, self.distribution_dec, (28,28,1), latent_dim=self.latent_dim)
        SVHN  = Modality('svhn' , svhn_enc,  svhn_dec, self.distribution_enc, self.distribution_dec, (32,32,3), latent_dim=self.latent_dim)
        Text  = Modality('text' , text_enc,  text_dec, self.distribution_enc, tfpl.OneHotCategorical, (8), latent_dim=self.latent_dim)

        self.modalities = {'mnist': MNIST, 'svhn': SVHN, 'text': Text} 
   
        self.modalities_params = MNIST.params + SVHN.params + Text.params

    def call(self, inputs):
        # both losses are dictionaries with 2^M-1 subsets
        kl_loss    = self.subsets_kl(inputs)
        recon_loss = self.subsets_reconstruction(inputs)
        weights = self.subsets_weights()
        
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
        
        self.codevae_loss = codevae_loss - 1000*H  
        losses['codevae_loss'] = codevae_loss
        losses['entropy'] = H
        return losses

    @tf.function
    def train(self, inputs, optimizer):
        with tf.GradientTape() as tape:
            losses = self.call(inputs)
        gradients = tape.gradient(self.codevae_loss, self.params)
        optimizer.apply_gradients(zip(gradients, self.params))

        return losses 
    
    @tf.function
    def train_hp(self, inputs, optimizer):
        with tf.GradientTape() as tape:
            losses = self.call(inputs)
            scaled_loss = optimizer.get_scaled_loss(self.codevae_loss)

        scaled_gradients = tape.gradient(scaled_loss, self.params)
        gradients = optimizer.get_unscaled_gradients(scaled_gradients)
        optimizer.apply_gradients(zip(gradients, self.params))
        
        return losses 
    
    @tf.function
    def evaluate(self, inputs, calculate_llik=False, batch_size=None, num_imp_samples=12,modality_specific=False, z_method='sample'):
        if calculate_llik:
            print('calculating log-likelihood...')
            # llik
            return self.loglikelihood(inputs,batch_size=batch_size,num_imp_samples=num_imp_samples)
        else:
            print('generating z reps...')
            all_z = self.latent_representations(inputs, z_method=z_method)
            print('generating modalities...')
            x_hat = self.generate_all_modalities(inputs, z_method=z_method)

            return all_z, x_hat
