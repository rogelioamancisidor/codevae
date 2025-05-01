from abc import ABC, abstractmethod
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import os
from itertools import chain, combinations
import tensorflow_probability as tfp
from CoDE import CoDE
from enc_dec import Weights

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

class CoDEVAE(ABC):
    # TODO make a list of available options for code_prior in argparse 
    def __init__(self, 
                 latent_dim,
                 prior=tfd.MultivariateNormalDiag,
                 correlation=0.5,
                 correlation_matrix=None,
                 code_prior = 'flat',
                 fusion_method = 'code',
                 distribution_fusion = tfd.MultivariateNormalDiag,
                 num_modalities=3,
                 name = 'codevae'
                ):
        super().__init__()
        self.prior  = prior(tf.zeros(latent_dim), tf.ones(latent_dim))
        self.latent_dim = latent_dim
        self.code = CoDE(corr=correlation, code_prior=code_prior, distribution_fusion=distribution_fusion, corr_matrix=correlation_matrix, modalities=num_modalities)
        self.fusion_method=fusion_method
        self.distribution_fusion = distribution_fusion
        print('CoDEVAE with fusion method {}'.format(fusion_method))
        
    @abstractmethod
    def modalities(self):
        # based on subsets_kl implementation
        # modalities is a dictionary of objects
        # The modality class must have a method
        # called qz, which is a distribution
        #raise NotImplementedError
        pass
    
    def weights_networks(self, trainable, n_subsets, init_val=100., given_probs=False):
        self.weights_nets = Weights(trainable=trainable, n_subsets=n_subsets,init_val=init_val,given_probs=given_probs)
        self.weights_nets_params = self.weights_nets.trainable_variables
        #self.weights_nets_params = self.weights_nets.variables
    
    def get_subsets(self):
        num_mods = len(list(self.modalities.keys()))

        xs = list(self.modalities)
        
        subsets_list = chain.from_iterable(combinations(xs, n) for n in range(len(xs)+1))
        subsets = dict()
        for k, mod_names in enumerate(subsets_list):
            mods = []
            for l, mod_name in enumerate(sorted(mod_names)):
                mods.append(self.modalities[mod_name])
            key = '_'.join(sorted(mod_names))
            subsets[key] = mods
       
        # XXX removing empty set 
        del subsets['']
        self.subsets = subsets

    # sum all Kullback-Leibler divergences for all subsets
    def fusion_code(self, subsets, inputs, num_imp_samples=None, training=True):
        subsets = subsets.split('_')
        mu_experts  = []
        std_experts = []
        for k in subsets:
            expert = self.modalities[k]
            x = inputs[k]
            if num_imp_samples is not None:
                x = tf.repeat(x, repeats=num_imp_samples, axis=0)

            qz = expert.qz_x
            qz_x = qz(x,training=training)
            
            mu_experts.append(qz_x.mean())
            std_experts.append(qz_x.stddev())
            
        # stacking so each row has all experts' assesments for the d'th dimension
        mu_experts_t  = tf.stack(mu_experts,axis=-1)
        std_experts_t = tf.stack(std_experts,axis=-1)

        # qz_x is a consensus distribution
        if self.fusion_method == 'code':
            qz_x = self.code.code_distribution(mu_experts_t, std_experts_t, subsets)
        elif self.fusion_method == 'poe':
            qz_x = self.code.poe_distribution(mu_experts_t, std_experts_t)
        return qz_x
    
    def rec_weights(self, norm_modality,norm_factor=1,rec_weights=None):
        if rec_weights is None:
            norm_mod_size = self.modalities[norm_modality].numel/norm_factor
            rec_weights = dict()
            for m in self.modalities.keys():
                mod = self.modalities[m]
                rec_weights[m] = norm_mod_size/mod.numel
        print('scaling weights on recon terms:',rec_weights)
        self.rec_weights = rec_weights

    # @inputs is a dictionary of data, keys are as in self.subsets
    def subsets_kl(self, inputs):
        kl_loss = dict()
        
        # all experts and consensus distributions
        for k, subsets in self.subsets.items():
            if len(subsets) == 1:
                expert = self.modalities[k]
                x = inputs[k] 
                qz = expert.qz_x
                qz_x = qz(x)
            elif len(subsets)>1:
                # using code to estimate posterior distributions
                # for subsets with cardinality > 1
                qz_x = self.fusion_code(k,inputs)
            
            kl   = tfd.kl_divergence(qz_x, self.prior)
            kl_loss[k] = kl
        
        return kl_loss
    
    # @inputs is a dictionary of data, keys are as in self.subsets
    def style_kl(self, inputs):
        kl_loss = dict()
        
        for m in self.modalities.keys():
            x = inputs[m] 
            style_expert = self.modalities[m]
            qs_x = style_expert.qz_x(x,modality_specific=True)
            
            # I simply take KL outside if  
            kl   = tfd.kl_divergence(qs_x, self.prior)
            kl_loss[m] = kl
        
        return kl_loss
    
    # sum all reconstructions for all subsets
    # modalities is a dictionary with data modalities
    # keys musy be like in self.subsets
    def subsets_reconstruction(self, inputs, modality_specific=False):
        recon_loss = dict()
        
        # all experts and consensus distributions
        for k, subsets in self.subsets.items():
            if len(subsets) == 1:
                x = inputs[k] 
                expert = self.modalities[k]
                qz_x = expert.qz_x(x)
            elif len(subsets)>1:
                qz_x = self.fusion_code(k,inputs)
            
            # sample z from subset
            z = qz_x.sample()
           
            recon_modalities=0
            for i,m in enumerate(self.modalities.keys()):
                x = inputs[m] 
                px = self.modalities[m].px_z
               
                if modality_specific:
                    if i == 0:
                        c = tf.identity(z)
                    style_expert = self.modalities[m]
                    qs_x = style_expert.qz_x(x, modality_specific=modality_specific)
                    s = qs_x.sample()
                    z = tf.concat([c,s],axis=1)
                
                px_z = px(z)
                
                # needed to handle both Laplace, Gaussian, and Bernoulli likelihoods
                if len(px_z.log_prob(x).get_shape().as_list()) > 1:
                    log_prob = tf.reshape(px_z.log_prob(x),[-1,self.modalities[m].numel])
                else:
                    log_prob = px_z.log_prob(x)

                recon = self.rec_weights[m]*log_prob
                recon_modalities+=tf.reduce_sum(recon,-1)
            
            recon_loss[k] = recon_modalities
        
        del z,x
        if modality_specific:
            del c,s
        return recon_loss 

    def subsets_weights(self):
        weights = dict()
        num_subsets = len(self.subsets)

        # get categorical pdf for weights
        p =  self.weights_nets([])
        w = p.probs
        
        for i,k in enumerate(self.subsets.keys()):
            weights[k] = w[i] 
        
        # calculate entropy
        H = p.entropy()
        weights['H'] = H 
        
        return weights
    
    
    #### XXX Methods for downstream tasks XXX ####
    # @from_modality is a string like in self.subsets. it can be something like mnist_svhn.
    # @to_modality is a key string like in self.subsets
    # XXX used for cross-modal generation
    def modality_generation(self, from_modality, to_modality, inputs, training=False, modality_specific=False):
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
            z = qz_x.sample()
        else:
            # generate modalities from prior
            for k in inputs.keys():
                pass
            z = self.prior.sample(inputs[k].shape[0])
        
        if modality_specific:
            if to_modality in from_modality.split('_'):
                style_expert = self.modalities[to_modality]
                x = inputs[to_modality]
                qs_x = style_expert.qz_x(x, modality_specific=modality_specific, training=training)
                s = qs_x.sample()
            else:
                s = self.prior_style.sample(z.shape[0])

            z = tf.concat([z,s],axis=1)
       
        px   = self.modalities[to_modality].px_z
        px_z = px(z, training=training)

        return px_z

    # generate latent representations z for all subsets 
    # @data is a TF dataset obj
    def latent_representations(self, inputs, z_method='mean', training=True):
        all_z = dict()
        for k, subsets in self.subsets.items():
            if len(subsets) == 1:
                x = inputs[k] 
                expert = self.modalities[k]
                qz_x = expert.qz_x(x,training=training)
            elif len(subsets)>1:
                qz_x = self.fusion_code(k,inputs,training=training)
            
            all_z[k] = getattr(qz_x,z_method)() 
        
        return all_z 
    
    # generate all modalites based on all subsets
    # @data is a TF dataset obj
    # XXX used for FIDs
    def generate_all_modalities(self, inputs, subsets=None, training=True, modality_specific=False, z_method='sample'):
        all_x = dict()
        
        if subsets is None:
            all_subsets = self.subsets.items()
        else:
            all_subsets = subsets

        if len(all_subsets)>1:
            for k, subsets in all_subsets: # this loop is to load experts
                x_hat_modalities = dict()
                for i,m in enumerate(self.modalities.keys()):    # this loop is gen modality
                    px = self.modalities[m].px_z
                    if len(subsets) == 1:
                        x = inputs[k] 
                        expert = self.modalities[k]
                        qz_x = expert.qz_x(x,training=training)
                    elif len(subsets)>1:
                        qz_x = self.fusion_code(k,inputs,training=training)
                    
                    z = getattr(qz_x,z_method)()
                    
                    if modality_specific:
                        if i == 0:
                            c = tf.identity(z)
                        if m in k.split('_'):
                            style_expert = self.modalities[m]
                            x = inputs[m] 
                            qs_x = style_expert.qz_x(x,training=training,modality_specific=modality_specific)
                            s = qs_x.sample()
                        else:
                            s = self.prior_style.sample(z.shape[0])
                        z = tf.concat([c,s],axis=1)
                    
                    px_z = px(z, training=training)
                    x_hat = px_z.mean()
                    x_hat_modalities[m] = x_hat
                all_x[k] = x_hat_modalities
            
        # generate modalities from prior
        x_hat_modalities = dict()
        z = self.prior.sample(inputs[m].shape[0])
        if modality_specific:
            z = tf.concat([z,self.prior_style.sample(inputs[m].shape[0])],axis=1)

        for m in self.modalities.keys():    
            px = self.modalities[m].px_z
            
            px_z = px(z,training=training)
            x_hat_modalities[m] = px_z.mean() 
        
        all_x['prior'] = x_hat_modalities

        return all_x

    def loglikelihood(self, inputs, batch_size=256, num_imp_samples=12, modality_specific=False):
        # XXX -------------------------------------------------------- XXX #
        def log_mean_exp(x, dim=-1):
            m = tf.reduce_max(x, axis=dim, keepdims=True)

            return m + tf.math.log(tf.reduce_mean(tf.math.exp(x - m), axis=dim, keepdims=True))

        def log_joint_estimate(prior, posterior, x_batch, z, batch_size, n_samples, modality_specific=False):
            # likelihoods and images are dicts with all subsets
            log_px_zs = 0
            

            for k, m_key in enumerate(self.modalities.keys()):
                px = self.modalities[m_key].px_z
                x = x_batch[m_key]
                if modality_specific:
                    qs_x = self.modalities[m_key].qz_x(x, modality_specific=modality_specific)
                    style_batch, _ = importance_sampling_z(qs_x, num_imp_samples)
                    likelihood = px(tf.concat([z,style_batch],axis=1),training=False)
                else:
                    likelihood = px(z)
                x = importance_sampling_x(x,n_samples)
                
                d_shape = likelihood.log_prob(x).get_shape().as_list()
                if len(d_shape) > 1:
                    likhood_log_prob = tf.reduce_sum(tf.reshape(likelihood.log_prob(x),[-1,self.modalities[m_key].numel]),-1)
                else:
                    likhood_log_prob = likelihood.log_prob(x)
                log_px_zs += likhood_log_prob

            log_weight_2d = log_px_zs + prior.log_prob(z) - posterior.log_prob(z) 
            log_weight    = tf.reshape(log_weight_2d,[batch_size, n_samples])

            # compute normalization constant for weights
            log_p = log_mean_exp(log_weight)
            
            return tf.reduce_mean(log_p)
        
        def log_marginal_estimate(likelihood, prior, posterior, x, z, batch_size, n_samples, m_key):
            d_shape = likelihood.log_prob(x).get_shape().as_list()

            if len(d_shape) > 1:
                likhood_log_prob = tf.reduce_sum(tf.reshape(likelihood.log_prob(x), [-1,self.modalities[m_key].numel]),-1)
            else:
                likhood_log_prob = likelihood.log_prob(x)

            log_weight_2d = likhood_log_prob + prior.log_prob(z) - posterior.log_prob(z) 
            

            log_weight    = tf.reshape(log_weight_2d,[batch_size, n_samples])

            # compute normalization constant for weights
            log_p = log_mean_exp(log_weight)
            
            return tf.reduce_mean(log_p)
        
        def calc_log_likelihood_batch(prior, posterior, x_batch, z_batch, batch_size, num_imp_samples, modality_specific=False):
            ll = dict();
            
            for k, m_key in enumerate(self.modalities.keys()):
                x = x_batch[m_key]
                px = self.modalities[m_key].px_z
                if modality_specific:
                    qs_x = self.modalities[m_key].qz_x(x, modality_specific=modality_specific)
                    style_batch, _ = importance_sampling_z(qs_x, num_imp_samples)
                    z = tf.concat([z_batch,style_batch],axis=1)
                    likelihood = px(z, training=False) #px_z
                else:
                    likelihood = px(z_batch) #px_z
                x = importance_sampling_x(x,num_imp_samples)
                
                # compute marginal log-likelihood
                ll_mod = log_marginal_estimate(likelihood, prior, posterior,
                                               x,
                                               z_batch,
                                               batch_size,
                                               num_imp_samples,
                                               m_key
                                               )
                ll[m_key] = ll_mod
            ll_joint = log_joint_estimate(prior, posterior,
                                          x_batch,z_batch,
                                          batch_size,
                                          num_imp_samples,
                                          modality_specific=modality_specific
                                          )
            ll['joint'] = ll_joint
            return ll
        # XXX -------------------------------------------------------- XXX #
       
        lhoods = dict()
        # make placeholders
        for k, s_key in enumerate(self.subsets.keys()):
            lhoods[s_key] = dict()
            for m, m_key in enumerate(self.modalities.keys()):
                lhoods[s_key][m_key] = {}
            lhoods[s_key]['joint'] = {}
       
        llik = {}
        for s_key, subsets in self.subsets.items():
            if len(subsets) == 1:
                expert = self.modalities[s_key]
                x = inputs[s_key]
                qz = expert.qz_x
                qz_x = qz(x)
            elif len(subsets)>1:
                qz_x = self.fusion_code(s_key,inputs)

            z_b, qz_x = importance_sampling_z(qz_x, num_imp_samples)
           
            ll_batch = calc_log_likelihood_batch(self.prior, qz_x,
                                                 inputs, z_b, batch_size,
                                                 num_imp_samples=num_imp_samples,modality_specific=modality_specific)

            for l, m_key in enumerate(ll_batch.keys()):
                lhoods[s_key][m_key] = ll_batch[m_key]
        return lhoods

def importance_sampling_z(qz_x, num_impsamples=12, distribution=tfd.MultivariateNormalDiag):
    mu = qz_x.mean()
    std = qz_x.stddev()
    N = mu.shape[0]
    d = mu.shape[1]

    mu  = tf.expand_dims(mu,0)
    std = tf.expand_dims(std,0)

    mu_impsamples  = tf.repeat(mu, num_impsamples, 0)
    std_impsamples = tf.repeat(std, num_impsamples, 0)

    eps = tf.random.normal([num_impsamples,N,d])
    z = mu_impsamples + eps*std_impsamples
            
    z = tf.reshape(z,[num_impsamples*N,-1])
    qz_x = distribution(tf.reshape(mu_impsamples,[num_impsamples*N,-1]),tf.reshape(std_impsamples,[num_impsamples*N,-1]))
    
    return z, qz_x
    
def importance_sampling_x(x, num_impsamples=12):
    d_shape = x.get_shape().as_list()
    N = x.shape[0]
    d = x.shape[1]

    if len(d_shape) == 3:
        x = tf.expand_dims(x,0)
        x = tf.repeat(x, num_impsamples, 0)
        x = tf.reshape(x, [N*num_impsamples, d_shape[-2], d_shape[-1]]) 
    elif len(d_shape) == 4:
        x = tf.expand_dims(x,0)
        x = tf.repeat(x, num_impsamples, 0)
        x = tf.reshape(x, [N*num_impsamples, d_shape[-3], d_shape[-2], d_shape[-1]]) 
    return x
