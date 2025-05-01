from matplotlib import pyplot as plt
import os
import numpy as np
import json
import random
import tensorflow as tf
import pickle

def transform(imgs, height=28, width=None):
    if width is None:
        width=height
    if height!= 64:
        imgs = tf.image.resize(imgs, [height,width], method=tf.image.ResizeMethod.BICUBIC)
    imgs = tf.clip_by_value(255*imgs, clip_value_min=0, clip_value_max=255).numpy().astype(np.uint8)
    
    return imgs 

def text_grid(text):
  """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
  # Create a figure to contain the plot.
  figure = plt.figure(figsize=(10,10))
  for i in range(100):
    # Start next subplot.
    plt.subplot(10, 10, i + 1)
    plt.subplots_adjust(hspace=0)
    plt.subplots_adjust(wspace=0)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    #plt.annotate(text[i],xy=(0.4,0.4),size=10,rotation=0)
    plt.annotate(text[i],xy=(0.1,0.4),size=10)

  return figure

def image_grid(images,cmap=None):
  """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
  # Create a figure to contain the plot.
  figure = plt.figure(figsize=(10,10))
  for i in range(100):
    # Start next subplot.
    plt.subplot(10, 10, i + 1)
    plt.subplots_adjust(hspace=0)
    plt.subplots_adjust(wspace=0)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images[i],cmap=cmap)

  return figure

class ConditionalGeneration:
    def __init__(self, data_folder='../data', dataset='MST', min_occ=3, max_sequence_length=32, vocab_file='cub.vocab'):
        if dataset == 'MST':
            with open(os.path.join(data_folder,'alphabet.json')) as alphabet_file:
                alphabet = str(''.join(json.load(alphabet_file)))
            
            self.alphabet = alphabet
        elif dataset == 'CUB':
            gen_dir = os.path.join(data_folder, "oc_{}_msl_{}".
                                     format(min_occ, max_sequence_length))
            with open(os.path.join(gen_dir, vocab_file), 'r') as vocab_file:
              vocab = json.load(vocab_file)
            self.i2w = vocab['i2w']


    def mst(self, from_modality, to_modality, data, model=None, name=None, output_folder='output', epoch=None, N=7, method='mean', grid_mode=False):
        text, mnist, svhn = data
        if grid_mode:
            N = 100
        idx_rnd = random.sample(range(mnist.shape[0]),N)

        mnist = mnist[idx_rnd,...]
        text  = text[idx_rnd,...]
        svhn  = svhn[idx_rnd,...]
        
        data = {'mnist':mnist, 'text':text, 'svhn':svhn}
        
        if model is not None:
            px_z = model.modality_generation(from_modality, to_modality, data)
            if to_modality == 'svhn':
                svhn = getattr(px_z,method)()
                svhn = transform(svhn)
                images=svhn
            elif to_modality == 'mnist':
                mnist = getattr(px_z,method)()
                images=mnist
            elif to_modality == 'text':
                text = getattr(px_z,method)()

        # convert back to original form
        text = self.retrive_text(text)
        
        if name is None and epoch is None:
            name = to_modality+"-"+from_modality
        elif name is None and epoch is not None:
            name = to_modality+"-"+from_modality+'_'+epoch

        if grid_mode:
            if to_modality != 'text':
                if to_modality == 'mnist':
                    cmap='gray'
                else:
                    cmap=None
                fig = image_grid(images,cmap)
            else:
                fig = text_grid(text)
            plt.savefig(os.path.join('..',output_folder,name+'.png'), format='png', bbox_inches='tight')
            plt.close(fig)
        else:
            self.plot_mst_images(text, svhn, mnist, name, output_folder)

    def retrive_text(self, x_texts):
        text_strings = []
        for i in range(x_texts.shape[0]):
            x_text = x_texts[i]
            text_string = ' '.join(map(str,[self.alphabet[i] for i in np.argmax(x_text,1)])).strip()
            text_strings.append(text_string)

        return text_strings
    
    def plot_mst_images(self, text, svhn, mnist, name, output_folder='output'):
        fig, axes = plt.subplots(3, 7, gridspec_kw = {'wspace':-0.01, 'hspace':-0.69})

        for i, ax in enumerate(axes.flat):
            if i < 7:
                ax.annotate(text[i],xy=(0.4,0.4),size=10,rotation=90)
                ax.axis('off')
            elif i < 14:
                ax.imshow(mnist[i-7,:,:],cmap='gray')
                ax.axis('off')
            else:
                ax.imshow(svhn[i-14,:,:])
                ax.axis('off')

        plt.savefig(os.path.join('..',output_folder,name+'.png'), format='png',bbox_inches="tight")
        plt.close('all')
