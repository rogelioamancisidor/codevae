import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
import json
import os
import pickle
from datetime import timedelta
import pandas as pd
import wandb

# supress all tf messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow.keras import mixed_precision

from CoDEVAE_MST import CoDEVAE_MST as CoDE_VAE
from NN_ClsImgs import ClsMNIST, ClsText, ClsSVHN
from enc_dec import EncMNIST_MLP as EncMNIST, DecMNIST_MLP as DecMNIST, EncSVHN, DecSVHN, EncText, DecText
from eval_metrics.ClassificationAcc import ClassificationAccuracy
from eval_metrics.CoherenceAcc import CoherenceAccuracy
from eval_metrics.FiDscore import FIDscores
from utils import load_mnist_svhn_text, save_generated_samples, sample_xhat_zte

def train():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--latent_dim",default= 20,help="dimensionality of the latent space", type=int)
    parser.add_argument("--epochs",default= 120, help="no. of epochs", type=int)
    parser.add_argument("--correlation",default= 0.4, help="correlation among experts' assessments", type=float)
    parser.add_argument("--batch_size",default= 256, help="batch size", type=int)
    parser.add_argument("--outfile",default="mnist_svhn_text",help="name to group runs in wandb", type=str)
    parser.add_argument("--fusion_method",default="code",choices=["poe",'code'],help="method to estimate consensus distributions", type=str)
    parser.add_argument("--distribution_enc",default="tfd.MultivariateNormalDiag", help="distribution for all q(z|x_k)")
    parser.add_argument("--distribution_dec",default="tfd.Laplace",help="distribution for p(x|z)")
    parser.add_argument("--prior",default="tfd.MultivariateNormalDiag", help="prior distribution for z")
    parser.add_argument('--wandb_mode',default='disabled',choices=['online','offline','disabled'],help='whether to log run')
    parser.add_argument('--wandb_name',default=None,help='short name to identify run')
    parser.add_argument("--save_every",default= 121, help="no. of epochs to save the model", type=int)
    parser.add_argument("--eval_every",default= 121, help="no. of epochs to run downstream evaluation tests", type=int)
    parser.add_argument("--create_dset",action='store_true', help="whether to create train dset")
    parser.add_argument("--subsets_trainable_weights", action='store_true', help="whether to train subsets' weights")
    parser.add_argument("--data_folder", default="../data", help="folder in which data sets are stored")
    parser.add_argument("--beta_kl", default=20, help="beta scaling factor for KL divergences", type=float)
    args = parser.parse_args()
    print (args)
   
    # you can pass the name argument to identify each run, otherwise it is random
    wandb.init(project='codevae_mst', group=args.outfile, mode=args.wandb_mode, name=args.wandb_name)
    wandb.run.log_code("./", include_fn=lambda path: path.endswith(".py"))
    wandb.config.update(args)
    print('wandb id {}'.format(wandb.run.id))
    print("Num GPUs:", len(tf.config.list_physical_devices('GPU')))
    
    # define output folder
    if args.wandb_mode!='disabled':
        output_folder = os.path.join(wandb.run.dir,'output')
    else:
        output_folder = './output'
    
    try:
        os.mkdir(output_folder)
        wandb.save('output')
    except OSError:
        print("Creation of the directory %s failed" % output_folder)
    else:
        print("Successfully created the directory %s" % output_folder)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    optimizer = mixed_precision.LossScaleOptimizer(optimizer)
    
    loss_codevae = tf.keras.metrics.Mean()
    mnist_kl    = tf.keras.metrics.Mean()
    mnist_recon = tf.keras.metrics.Mean()
    svhn_kl     = tf.keras.metrics.Mean()
    svhn_recon  = tf.keras.metrics.Mean()
    text_kl     = tf.keras.metrics.Mean()
    text_recon  = tf.keras.metrics.Mean()
    entropy     = tf.keras.metrics.Mean()

    # load alphabet
    with open(os.path.join(args.data_folder,'alphabet.json')) as alphabet_file:
        alphabet = str(''.join(json.load(alphabet_file)))

    # load tr dset
    if args.create_dset:
        print('creating dset. wait....')
        tr_mnist, tr_svhn, tr_text, tr_target, te_mnist, te_svhn, te_text, te_target = load_mnist_svhn_text(alphabet)
        train_data_nn = {'mnist':tr_mnist, 'svhn':tr_svhn, 'text':tr_text, 'target':tr_target}

        idx  = random.sample(range(int(tr_target.shape[0])),int(500))
        train_data_nn_small = {k: d[idx,...] for k, d in train_data_nn.items()}
        print('saving 500 obs from trainig data..')
        with open(os.path.join(args.data_folder,"train_mst_"+str(wandb.run.id)+".pkl"), 'wb') as f:
            pickle.dump(train_data_nn_small, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        tr_dict = {'text':tr_text,'mnist':tr_mnist,'svhn':tr_svhn} 
        te_dict = {'text':te_text,'mnist':te_mnist,'svhn':te_svhn} 
        BUFFER  = tr_mnist.shape[0]
        del train_data_nn, train_data_nn_small, tr_mnist, tr_svhn, tr_text, te_mnist, te_svhn, te_text
    else:
        print('train data already exists! loading it...')
        with open(os.path.join(args.data_folder,"train_mst_"+str(wandb.run.id)+".pkl"), 'rb') as f:
            train_data_nn = pickle.load(f)
        tr_mnist  = train_data_nn['mnist']
        tr_svhn   = train_data_nn['svhn']
        tr_text   = train_data_nn['text']
        tr_target = train_data_nn['target']
        
        print('test data already exists! loading it...')
        with open(os.path.join(args.data_folder,"test_mst_"+str(wandb.run.id)+".pkl"), 'rb') as f:
            test_data_nn = pickle.load(f)
        te_mnist  = test_data_nn['mnist']
        te_svhn   = test_data_nn['svhn']
        te_text   = test_data_nn['text']
        te_target = test_data_nn['target']
    print('tr size {} and te size {}'.format(tr_target.shape[0], te_target.shape[0]))
        
    print('building model...')
    # NOTE all encs & decs use default values
    model = CoDE_VAE(args.latent_dim,
                    EncMNIST, DecMNIST,
                    EncSVHN, DecSVHN,
                    EncText, DecText,
                    correlation=args.correlation,
                    beta=args.beta_kl,
                    fusion_method=args.fusion_method,
                    prior=eval(args.prior),
                    distribution_enc=eval(args.distribution_enc),
                    distribution_dec=eval(args.distribution_dec),
                    subsets_trainable_weights = args.subsets_trainable_weights
                    )
    trainableParams = np.sum([np.prod(v.get_shape()) for v in model.weights])
    print('CoDEVAE model has {} trainable params'.format(trainableParams))
    
    # build evaluation metric objects
    coherence_acc = CoherenceAccuracy(modality_cls={'mnist':ClsMNIST,'svhn':ClsSVHN,'text':ClsText})
    
    # ckpts
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, model=model) 
    manager    = tf.train.CheckpointManager(checkpoint, os.path.join(output_folder,"ckpts"), max_to_keep=3)
   
    tr_data = tf.data.Dataset.from_tensor_slices(tr_dict).shuffle(BUFFER).batch(args.batch_size)
    del tr_dict
    
    print('training...')
    # time it
    start = time.time()
    all_fids = dict() 
    all_acc = dict()
    all_coherence=dict()
    all_batch = []
    while int(checkpoint.step) < args.epochs:
        for i, data_batch in enumerate(tr_data):
            losses = model.train_hp(data_batch,optimizer)

            loss_codevae(losses['codevae_loss']) # this saves the average loss during training
            mnist_kl(losses['mnist']['kl']) 
            mnist_recon(losses['mnist']['recon']) 
            svhn_kl(losses['svhn']['kl']) 
            svhn_recon(losses['svhn']['recon']) 
            text_kl(losses['text']['kl']) 
            text_recon(losses['text']['recon']) 
            entropy(losses['entropy'])
       
        # XXX START wandb logging
        # plot weights
        fig, ax = plt.subplots() 
        weights = [losses['mnist']['weight'].numpy(),losses['svhn']['weight'].numpy(), 
                   losses['text']['weight'].numpy(),losses['mnist_svhn']['weight'].numpy(), 
                   losses['mnist_text']['weight'].numpy(), losses['svhn_text']['weight'].numpy(), 
                   losses['mnist_svhn_text']['weight'].numpy()
                   ]
        weights_labels=['mnist','svhn','text','mnist_svhn','mnist_text','svhn_text','mnist_svhn_text']
        ax.bar(list(range(len(weights_labels))),weights)
        for i,l in enumerate(weights_labels):
            ax.annotate(l, xy=(i,weights[i]+0.003), ha='center', va='bottom')
        
        # metrics
        wandb.log({'loss':loss_codevae.result(),
                   'mnist_kl':mnist_kl.result(), 
                   'mnist_recon':mnist_recon.result(), 
                   'svhn_kl':svhn_kl.result(), 
                   'svhn_recon':svhn_recon.result(), 
                   'text_kl':text_kl.result(), 
                   'text_recon':text_recon.result(),
                   'entropy':entropy.result(),
                   'weights_plot': fig,
                  })
        plt.close()
        # XXX END  wandb logging
        
        if (int(checkpoint.step)+1) % args.eval_every == 0:
            print('epoch {:04d}: loss is {:0.4f}'.format(int(checkpoint.step)+1,losses['codevae_loss']))
           
            # XXX you can add other downstream tasks inhere. see test file. 
            all_coherence_dicts=[]
            xte_hat,_ = sample_xhat_zte(model, te_dict, args, z_method='mean')
            print('evaluating coherence acc...')
            coherence_subsets,cross_coherence = coherence_acc.get_coherence(xte_hat, te_target)
            all_coherence_dicts.append(cross_coherence)
            user_dict = pd.DataFrame(all_coherence_dicts)
            all_coherence = pd.DataFrame.from_dict({(i,j): user_dict[i][j] 
                                                    for i in user_dict.keys() 
                                                    for j in user_dict[i].keys()},
                                                    orient='index')
            print('------------- Average coherence -------------')
            print(all_coherence.groupby(level=0).mean())
            print('averall coherence {:.4f}'.format(all_coherence.mean(numeric_only=True).mean()))
            del xte_hat,all_coherence_dicts
        
        if (int(checkpoint.step)+1) % args.save_every == 0 or ((int(checkpoint.step)+1)==args.epochs):
            save_path = manager.save(checkpoint_number=int(checkpoint.step)+1)
            print("Saved checkpoint for step {}: {}".format(int(checkpoint.step)+1, save_path))
        
        # increse checkpoint  
        checkpoint.step.assign_add(1)
    
    print('elapsed time: {}'.format(timedelta(seconds=time.time()-start)))

    with open(os.path.join(output_folder,'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
if __name__ == "__main__":
    train()
