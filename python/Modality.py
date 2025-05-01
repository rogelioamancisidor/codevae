import numpy as np

class Modality():
    def __init__(self, name, enc, dec, enc_distribution, dec_distribution, data_size, latent_dim, modality_specific=False, latent_dim_s=None):
        self.name = name
        self.modality_specific = modality_specific
        if self.modality_specific:
            self.encoder = enc(distribution=enc_distribution, latent_dim=latent_dim, latent_dim_s=latent_dim_s, modality_specific=modality_specific)
            self.decoder = dec(distribution=dec_distribution, latent_dim=latent_dim, latent_dim_s=latent_dim_s, modality_specific=modality_specific)
        else:
            self.encoder = enc(distribution=enc_distribution, latent_dim=latent_dim)
            self.decoder = dec(distribution=dec_distribution, latent_dim=latent_dim)
        self.data_size = data_size

    def name(self):
        return self.name
    
    @property
    def numel(self):
        return np.prod(self.data_size)

    @property
    def params(self):
        #params  = self.encoder.trainable_variables + self.decoder.trainable_variables
        params  = self.encoder.variables + self.decoder.variables
        return params

    @property
    def qz_x(self):
        return self.encoder
    
    @property
    def px_z(self):
        return self.decoder

    def generative_model(self,x):
        qz_x = self.encoder(x)
        z   = qz_x.sample()

        px_z = self.decoder(z)
        
        return px_z

