import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from scipy import linalg
import tensorflow_gan as tfgan
import pathlib
import os
from tensorflow.keras.utils import image_dataset_from_directory as load_images

class FIDscores():
    def __init__(self):
        pass

    def calculate_fid(self, feats_real, feats_gen):
        mu_real = np.mean(feats_real, axis=0)
        sigma_real = np.cov(feats_real, rowvar=False)
        mu_gen = np.mean(feats_gen, axis=0)
        sigma_gen = np.cov(feats_gen, rowvar=False)
        fid = self.calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
        return fid;

    def from_file_activations(self, file_real, file_gen, batch_size=50):
        if not os.path.exists(file_real):
            raise RuntimeError('Invalid path: %s' % file_real)
        if not os.path.exists(file_gen):
            raise RuntimeError('Invalid path: %s' % file_gen)
    
        real_image_embeddings = np.load(file_real)
        generated_image_embeddings = np.load(file_gen)
        
        fid = self.calculate_fid(real_image_embeddings,generated_image_embeddings)
        return fid

    def from_file_png(self, file_real, file_gen, batch_size=256):
        path_real  = pathlib.Path(file_real)
        files_real = list(path_real.glob('*.jpg')) + list(path_real.glob('*.png'))
        path_gen  = pathlib.Path(file_gen)
        files_gen = list(path_gen.glob('*.jpg')) + list(path_gen.glob('*.png'))

        print('loading images...')
        ds_real = load_images(path_real,
                         labels=None,
                         color_mode='rgb',
                         image_size=(299,299),
                         batch_size=batch_size,
                         shuffle=False
                         )
        
        ds_gen = load_images(path_gen,
                         labels=None,
                         color_mode='rgb',
                         image_size=(299,299),
                         batch_size=batch_size,
                         shuffle=False
                         )

        fid = self.get_fid(ds_real, ds_gen, scale_0_1=True)
        return fid
    
    def from_array(self, real_images, gen_images, batch_size=256, size=299, print_out=False):
        num_chls = real_images.shape[-1]

        if print_out:
            print('resizing and converting to rgb...')
        ds_real = tf.data.Dataset.from_tensor_slices(real_images)
        ds_gen  = tf.data.Dataset.from_tensor_slices(gen_images)
        del gen_images, real_images
        
        AUTOTUNE = tf.data.AUTOTUNE
        
        # resize
        ds_real = ds_real.map(lambda img: tf.image.resize(img, [size,size]),num_parallel_calls=AUTOTUNE)
        ds_gen  = ds_gen.map(lambda img: tf.image.resize(img, [size,size]),num_parallel_calls=AUTOTUNE)

        # convert to rgb
        if num_chls == 1:
            ds_real = ds_real.map(lambda img: tf.image.grayscale_to_rgb(img))
            ds_gen  = ds_gen.map(lambda img: tf.image.grayscale_to_rgb(img))

        ds_real = ds_real.cache()
        ds_real = ds_real.batch(batch_size)
        ds_real = ds_real.prefetch(buffer_size=AUTOTUNE)

        ds_gen = ds_gen.cache()
        ds_gen = ds_gen.batch(batch_size)
        ds_gen = ds_gen.prefetch(buffer_size=AUTOTUNE)
        
        fid = self.get_fid(ds_real, ds_gen)

        return fid

    def get_fid(self, real_images, gen_images, scale_0_1=False, print_out=False):
        generated_image_embeddings=[]
        real_image_embeddings=[]
        
        dataset = tf.data.Dataset.zip((real_images, gen_images))                
       
        if print_out:
            print('getting inception activations...')

        for i, (real_images, gen_images) in enumerate(dataset):
            if scale_0_1:
                real_images/=255.
                gen_images/=255.

            # compute embeddings for real images
            real_image_embedding = self.get_activations(real_images)
            
            # compute embeddings for generated images
            generated_image_embedding = self.get_activations(gen_images)

            real_image_embeddings.extend(real_image_embedding)
            generated_image_embeddings.extend(generated_image_embedding)
        
        fid = self.calculate_fid(tf.stack(real_image_embeddings,axis=0), tf.stack(generated_image_embeddings,axis=0))

        return fid

    @tf.function
    def get_activations(self,images, num_splits=1, size=299):
        INCEPTION_TFHUB = 'https://tfhub.dev/tensorflow/tfgan/eval/inception/1'
        INCEPTION_FINAL_POOL = 'pool_3'
        
        # rescale pixels between [-1 1]
        images = 2.*images - 1.

        generated_images_list = tf.split(images, num_or_size_splits = num_splits)
        
        activations = tf.nest.map_structure(tf.stop_gradient, tf.map_fn(
                        fn = tfgan.eval.classifier_fn_from_tfhub(INCEPTION_TFHUB, INCEPTION_FINAL_POOL, True),
                        elems = tf.stack(generated_images_list),
                        parallel_iterations = 10,
                        swap_memory = True,
                        name = 'RunClassifier'))
        
        activations = tf.concat(tf.unstack(activations), 0)

        return activations

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.
        Returns:
        --   : The Frechet Distance.
        """
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                covmean = np.nan;
                print('Imaginary component {}'.format(m));
            else:
                covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)
