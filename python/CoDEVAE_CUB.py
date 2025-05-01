import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from CoDEVAE import CoDEVAE
from Modality import Modality
import tensorflow as tf
import tensorflow_probability as tfp
tfpl = tfp.layers
tfd = tfp.distributions
from tensorflow.keras import mixed_precision

class CoDEVAE_CUB(CoDEVAE, tf.keras.Model):
    def __init__(self, 
                 latent_dim, 
                 latent_dim_s, 
                 img_enc,  img_dec,
                 text_enc, text_dec,
                 correlation=0.5,
                 correlation_matrix=None,
                 prior=tfd.MultivariateNormalDiag, 
                 distribution_enc=tfd.MultivariateNormalDiag,
                 distribution_dec=tfd.Laplace,
                 code_prior='flat', 
                 fusion_method='code',
                 subsets_trainable_weights = True,
                 beta = 5.0,
                 beta_style_img = 1.0,
                 beta_style_txt = 1.0,
                 train=True,
                 rec_weights=None,
                 modality_specific=False,
                 given_weights=False,
                 **kwargs):
        CoDEVAE.__init__(self,latent_dim, prior=prior, fusion_method=fusion_method, distribution_fusion=distribution_enc, code_prior=code_prior, correlation=correlation, correlation_matrix=correlation_matrix)
        tf.keras.Model.__init__(self,**kwargs)
        self.latent_dim = latent_dim
        self.latent_dim_s = latent_dim_s
        self.distribution_enc = distribution_enc
        self.distribution_dec = distribution_dec
        self.modality_specific = modality_specific
        
        # initialize abstract method
        self.modalities(img_enc, img_dec, text_enc, text_dec)
        # call get_subsets
        self.get_subsets()
        # call rec_weights, pass modality with more no of params
        self.rec_weights('text',rec_weights=rec_weights)
        # beta to scale KL
        self.beta = beta
        self.beta_style = {'image':beta_style_img, 'text':beta_style_txt}
        # weights network
        self.subsets_trainable_weights = subsets_trainable_weights

        if given_weights:
            self.weights_networks(trainable=False, n_subsets=1, init_val=[0.25,0.325,0.425], given_probs=True)
        else:
            self.weights_networks(trainable=subsets_trainable_weights, n_subsets=len(self.subsets.keys()))
        
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

    def modalities(self, img_enc, img_dec, text_enc, text_dec):
        Image = Modality('image', img_enc, img_dec, self.distribution_enc, self.distribution_dec, (64,64,3), self.latent_dim, self.modality_specific, latent_dim_s=self.latent_dim_s)
        Text  = Modality('text' , text_enc,  text_dec, self.distribution_enc, tfpl.OneHotCategorical, (32), self.latent_dim, self.modality_specific, latent_dim_s=self.latent_dim_s)

        self.modalities = {'image': Image, 'text': Text} 
   
        self.modalities_params = Image.params + Text.params

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
        ms_kls = dict()
        if self.modality_specific:
            for m in self.modalities.keys():
                ms_kls[m] = tf.reduce_mean(kl_style[m])

                # XXX I use + KL as at the end we minimize -ELBO
                style_kl += weights[m]*(tf.reduce_mean(self.beta_style[m]*kl_style[m]))
        

        self.codevae_loss = codevae_loss + style_kl - 1000*H 
        losses['ms_kl'] = ms_kls
        losses['codevae_loss'] = codevae_loss
        return losses


    # XXX used for coherence
    def style_generation(self, from_modality, to_modality, inputs, training=False, fix_style=False, fix_content=False, N=10, N_rep=10, z_method='sample'):
        if from_modality!='prior':
            subsets = self.subsets[from_modality]
            if len(subsets) == 1:
                # recognition model
                qz   = self.modalities[from_modality].qz_x
                x    = inputs[from_modality]
                qz_x = qz(x,training=training)
            else:
                qz_x = self.fusion_code(from_modality, inputs, training=training)

            # generative model
            c = getattr(qz_x,z_method)()
        else:
            # generate modalities from prior
            for k in inputs.keys():
                pass
            c = getattr(self.prior,z_method)(inputs[k].shape[0])
        
        if fix_content:
            c = c[0:N, ...]
            c = tf.repeat(c,N_rep,axis=0)
        
        if self.modality_specific:
            if to_modality in from_modality.split('_'):
                style_expert = self.modalities[to_modality]
                x = inputs[to_modality]
                qs_x = style_expert.qz_x(x, modality_specific=self.modality_specific, training=training)
                s = qs_x.sample()
            else:
                for k in inputs.keys():
                    pass
                s = self.prior_style.sample(inputs[k].shape[0])

        if fix_style:
            s = s[0:N, ...]
            s = tf.repeat(s,N_rep,axis=0)

        if self.modality_specific:
            z = tf.concat([c,s],axis=1)
        else: 
            z = c
        
        px   = self.modalities[to_modality].px_z
        px_z = px(z, training=training)

        return px_z

    @tf.function
    def train(self, inputs, optimizer):
        with tf.GradientTape() as tape:
            losses = self.call(inputs)
        gradients = tape.gradient(self.codevae_loss, self.params.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.params.trainable_variables))

        return losses 
    
    @tf.function
    def train_hp(self, inputs, optimizer):
        with tf.GradientTape() as tape:
            losses = self.call(inputs)
            scaled_loss = optimizer.get_scaled_loss(self.codevae_loss)

        scaled_gradients = tape.gradient(scaled_loss, self.params.trainable_variables)
        gradients = optimizer.get_unscaled_gradients(scaled_gradients)
        optimizer.apply_gradients(zip(gradients, self.params.trainable_variables))
        
        return losses 
    
    @tf.function
    def evaluate(self, inputs, calculate_llik=False, batch_size=None, num_imp_samples=12, training=False, modality_specific=False, z_method='sample'):
        if calculate_llik:
            print('calculating log-likelihood...')
            # llik
            return self.loglikelihood(inputs,batch_size,num_imp_samples=num_imp_samples,modality_specific=modality_specific)
        else:
            print('generating z reps...')
            all_z = self.latent_representations(inputs,training=training,z_method=z_method)
            print('generating modalities...')
            x_hat = self.generate_all_modalities(inputs,training=training, modality_specific=modality_specific,z_method=z_method)

            return all_z, x_hat
