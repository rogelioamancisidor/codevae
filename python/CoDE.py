import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow_probability as tfp
import tensorflow as tf
tfd = tfp.distributions

class CoDE:
    def __init__(self, corr=0.5, code_prior='flat', distribution_fusion=tfd.MultivariateNormalDiag, corr_matrix=None, modalities = 4):
        self.corr = corr
        self.distribution = distribution_fusion
        # TODO code results for different priors 
        self.code_prior = code_prior

        # if corralation matrix is not given, assume common correlation given by corr
        if corr_matrix is None:
            corr_matrix = self.corr * tf.ones((modalities,modalities),dtype=tf.float32)
            corr_matrix = tf.linalg.set_diag(corr_matrix, tf.ones(modalities,dtype=tf.float32))
        self.corr_matrix = corr_matrix

    def experts_inv_covariance(self, std_experts):
        # Input size: (batch_size, zdim, experts)
        experts = std_experts.shape[2] 
        zdim    = std_experts.shape[1]
       
        # transpose to do calculations fully vectorized
        std_experts = tf.transpose(std_experts, perm=[1, 0, 2])

        std = std_experts[...,None]
        cov_matrix = tf.linalg.matmul(std,tf.transpose(std,perm=[0,1,3,2]))*self.corr_matrix[None,:experts,:experts]
        inv_cov_matrix = tf.linalg.inv(cov_matrix)
        all_experts_inv = inv_cov_matrix
        
        return all_experts_inv

    def get_code_params(self, mu_experts):
        sum_inv_cov = tf.reduce_sum(self.all_experts_inv, axis=(2,3))[...,None]
        # weights as in Winkler
        weights = tf.reduce_sum(self.all_experts_inv, axis=(3))/sum_inv_cov
        code_mu = tf.reduce_sum(tf.transpose(weights,perm=[1,0,2])*mu_experts,2)
        code_var = tf.transpose(tf.squeeze(1/sum_inv_cov))
        
        self.code_mus = code_mu
        self.code_variances = code_var

    def code_distribution(self, mu_experts, std_experts, eps=1e-8):
        self.all_experts_inv = self.experts_inv_covariance(std_experts)
        self.get_code_params(mu_experts)

        pdf = self.distribution(self.code_mus, tf.math.sqrt(self.code_variances))
        
        return pdf
    
    def poe_distribution(self, mu_experts, std_experts):
        var = std_experts**2
        precision = 1. / var
        poe_mus   = tf.reduce_sum(mu_experts*precision,2)/tf.reduce_sum(precision,2)
        poe_vars  = 1. / tf.reduce_sum(precision,2)
        
        pdf = self.distribution(poe_mus, tf.math.sqrt(poe_vars))
        
        return pdf
