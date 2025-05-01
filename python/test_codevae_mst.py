from CoDEVAE_MST import CoDEVAE_MST as CoDE_VAE
from NN_ClsImgs import ClsMNIST, ClsText, ClsSVHN
from enc_dec import EncMNIST_MLP as EncMNIST, DecMNIST_MLP as DecMNIST, EncSVHN, DecSVHN, EncText, DecText
from utils import load_mnist_svhn_text, save_generated_samples, sample_xhat_zte

from eval_metrics.FiDscore import FIDscores
from eval_metrics.ConditionalGeneration import ConditionalGeneration
from eval_metrics.ClassificationAcc import ClassificationAccuracy
from eval_metrics.CoherenceAcc import CoherenceAccuracy

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow_probability as tfp
tfd = tfp.distributions

import shutil
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
import wandb
import random
import time
import pandas as pd
import shutil
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# files to restore from wandb
restore_files = ['checkpoint','ckpt-.index','ckpt-.data-00000-of-00001','commandline_args.txt']

def load_model(args, i, folder_name='./output'):
    if args.restore_from_wandb:
        print('loading CoDEVAE model from wandb {}'.format(str(args.wandb_run[i])))
    else:
        print('loading CoDEVAE model from {}'.format(os.path.join(folder_name,args.ckpt_folder)))

    model = CoDE_VAE(args.latent_dim,
                    EncMNIST, DecMNIST,
                    EncSVHN, DecSVHN,
                    EncText, DecText,
                    correlation=args.correlation,
                    fusion_method=args.fusion_method,
                    prior=eval(args.prior),
                    distribution_enc=eval(args.distribution_enc),
                    distribution_dec=eval(args.distribution_dec),
                    subsets_trainable_weights = args.subsets_trainable_weights,
                    train=False
                    )

    # provide checkpoint and manager
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(checkpoint,os.path.join(folder_name,args.ckpt_folder),max_to_keep=1)
    
    # restore last model if exits
    #checkpoint.restore(manager.latest_checkpoint).expect_partial()
    # eventually specify a checkpoint. see bellow
    f = 'ckpt-.index'
    file_restore = args.restore_ckpt.join([f[:5], f[5:]])
    checkpoint.restore(os.path.join(folder_name,'ckpts/')+file_restore.split('.')[0]).expect_partial()
    if manager.latest_checkpoint:
        print("Model restored from {}".format(file_restore.split('.')[0]))
    else:
        print("Model {}  does not exist".format(file_restore.split('.')[0]))

    return model

def placeholder_llik(model):
    # make placeholders
    lhoods = dict()
    all_subsets = list(model.subsets.keys()) 
    for k, s_key in enumerate(all_subsets):
        lhoods[s_key] = dict()
        for m, m_key in enumerate(model.modalities.keys()):
            lhoods[s_key][m_key] = []
        lhoods[s_key]['joint'] = []
    return lhoods

def main(args):

    if args.calculate_fid:
        fids = dict()
        for fid_modality in args.fid_modalities:
            fids[fid_modality] = []
        all_fid_dicts = []

    if args.coherence_acc:
        all_coherence_dicts = []
        all_avg_cohe = []

    print("Num GPUs:", len(tf.config.list_physical_devices('GPU')))
    print('------------------ Evaluating config {} --------------------'.format(args.output_file))
    start = time.time()
    for i, wandb_run in enumerate(args.wandb_run):
        if args.restore_from_wandb:
            if os.path.exists('./output'):
                print('folder with pretrained weights already exists. Deleting it ...')
                shutil.rmtree('./output', ignore_errors=True)
            
            name_wandb_run = os.path.join(args.account_name, args.project_name, wandb_run)
            api = wandb.Api()
            # restore files 
            for j,f in enumerate(restore_files):
                if f == 'commandline_args.txt':
                    file = api.run(name_wandb_run).file(name=os.path.join('./output',f))
                elif j in(1,2):
                    f = args.restore_ckpt.join([f[:5], f[5:]])
                    file = api.run(name_wandb_run).file(name=os.path.join('./output',args.ckpt_folder,f))
                else:
                    file = api.run(name_wandb_run).file(name=os.path.join('./output',args.ckpt_folder,f))
                file.download(replace=True)
        else:
            assert os.path.exists(os.path.join('./output',args.ckpt_folder)),'ckpts must be downloaded first in the path {}'.format(os.path.join('./output',args.ckpt_folder))

        with open(os.path.join('./output','commandline_args.txt'), 'r') as f:
            args.__dict__.update(json.load(f))
        
        # load model
        model = load_model(args,i)
            
        if i == 0:
            print('loading test dset..')
            with open(os.path.join(args.data_folder,'alphabet.json')) as alphabet_file:
                alphabet = str(''.join(json.load(alphabet_file)))
            tr_mnist, tr_svhn, tr_text, tr_target, te_mnist, te_svhn, te_text, te_target = load_mnist_svhn_text(alphabet)
            
            te_dict = {'mnist':te_mnist, 'svhn':te_svhn, 'text':te_text, 'target':te_target}
            print('shape test dset {}'.format(te_dict['target'].shape[0]))
            te_data = tf.data.Dataset.from_tensor_slices(te_dict).batch(args.batch_size)
            
            # selection 10K randomly otherwise FID cannot be calculated
            idx  = random.sample(range(te_mnist.shape[0]),10000)
            te_dict_fid = {k: d[idx,...] for k, d in te_dict.items()}

            if args.calculate_loglik:
                all_llik = {'mnist':[],'svhn':[],'text':[],'mnist_svhn':[],'mnist_text':[],'svhn_text':[],'mnist_svhn_text':[]}
                tr_data = tf.data.Dataset.from_tensor_slices({'mnist':tr_mnist, 'svhn':tr_svhn, 'text':tr_text, 'target':tr_target}).batch(100)
            
            if args.lr_acc:
                # dict to save all acc
                self_acc  = dict()
                cross_acc = dict()
                for subset in model.subsets.keys():
                    self_acc[subset] = []
                    cross_acc[subset] = []

            if args.coherence_joint:
                coherence_all = []
        
        if args.lr_acc:
            print('evaluation classification performance...')
            print('loading train dset..')
            # used 500 obs from train set to train linear cls
            with open(os.path.join(args.data_folder,"train_mst_"+str(wandb_run)+".pkl"), 'rb') as f:
                train_data_nn = pickle.load(f)
            
            tr_mnist  = train_data_nn['mnist']
            tr_svhn   = train_data_nn['svhn']
            tr_text   = train_data_nn['text']
            tr_target = train_data_nn['target']
            print('training dset shape', tr_mnist.shape)

            tr_dict = {'mnist':tr_mnist, 'svhn':tr_svhn, 'text':tr_text, 'target':tr_target}
            tr_data = tf.data.Dataset.from_tensor_slices(tr_dict).batch(500)
            
            for i, data_batch in enumerate(tr_data):
                z_tr,_ = model.evaluate(data_batch)
                break

        
        if args.calculate_trace:
            print('------------- Trace for all subsets --------------')
            for k, subsets in model.subsets.items():
                print('calculating trace of subset {}'.format(k))
                if len(subsets) == 1:
                    x = te_dict[k] 
                    expert = model.modalities[k]
                    qz_x = expert.qz_x(x, training=False)
                elif len(subsets)>1:
                    qz_x = model.fusion_code(k, te_dict, training=False)

                # sample std
                std = qz_x.stddev()
                var = tf.math.square(std)
                trace = tf.reduce_sum(var,axis=1)
                avg_trace = tf.reduce_mean(trace)

                print('trace for subset {} is {:.4f}'.format(k,avg_trace.numpy()))

        
        if args.lr_acc:
            cls_acc = ClassificationAccuracy()
            _,z_te = sample_xhat_zte(model, te_dict, args, z_method='mean')
            
            print('evaluatin lr acc..')
            tr_target = data_batch['target']

            acc_subsets, acc_cross, acc_self = cls_acc.get_accuracy(tr_target, te_dict['target'], z_tr, z_te)

            for k1 in acc_cross.keys():
                temp_d = acc_cross[k1]
                for k2, acc in temp_d.items():
                    cross_acc[k1].append(acc)
            for k1 in acc_self.keys():
                self_acc[k1].append(acc_self[k1])

        if args.coherence_acc:
            coherence_acc = CoherenceAccuracy(modality_cls={'mnist':ClsMNIST,'svhn':ClsSVHN,'text':ClsText})
            print('evaluating coherence acc..')
            xhat_te,_ = sample_xhat_zte(model, te_dict, args, z_method='mean')
            
            coherence_subsets,  cross_coherence = coherence_acc.get_coherence(xhat_te, te_dict['target'])

            # save individual runs to calculate std for overall coherence
            temp_dict = pd.DataFrame([cross_coherence])
            temp_coherence = pd.DataFrame.from_dict({(i,j): temp_dict[i][j]
                                                    for i in temp_dict.keys()
                                                    for j in temp_dict[i].keys()},
                                                    orient='index')
            all_avg_cohe.append(temp_coherence.mean(numeric_only=True).mean())

            # save to list to calculate cross-modal coherence overview
            all_coherence_dicts.append(cross_coherence)
            
        if args.coherence_joint:
            coherence_acc = CoherenceAccuracy(modality_cls={'mnist':ClsMNIST,'svhn':ClsSVHN,'text':ClsText})
            print('evaluating joint coherence...')
            xhat_te,_ = sample_xhat_zte(model, te_dict, args, z_method='mean')
            
            joint = coherence_acc.joint_coherence(xhat_te, te_dict['target'])

            coherence_all.append(joint)

        if args.crossmodal_generation:
            te_data = (te_text, te_mnist, te_svhn)
            crossmodal_gen = ConditionalGeneration()

            if args.from_modalities[0] == 'all':
                from_modalities = model.subsets.keys()
            else:
                from_modalities = args.from_modalities

            for from_modality in from_modalities:
                for to_modality in args.to_modalities:
                    print('generating {} image conditioned on {}'.format(to_modality,from_modality))
                    crossmodal_gen.mst(from_modality, to_modality, te_data, model=model, method='mean', grid_mode=args.grid_mode)

        if args.calculate_fid:
            print('exaluation FID scores...')
            # create FID object
            fid_score = FIDscores()
            # XXX the folder /tmp/fid must exist!
            # this is a must if code runs on a server!
            
            xhat_te, _ = sample_xhat_zte(model, te_dict_fid, args, z_method='sample')
            save_generated_samples(xhat_te, te_dict_fid, args.save_gen_imgs)
            
            if args.from_subsets[0]=='all':
                from_subsets = list(model.subsets.keys()) + ['prior']
            else:
                from_subsets = args.from_subsets
            
            cond_fid = dict()
            for from_subset in from_subsets:
                print('evaluating FID from subset {}...'.format(from_subset))
                mods_fic = dict()
                for fid_modality in args.fid_modalities:
                    file_real = os.path.join(args.save_gen_imgs,'real',fid_modality)
                    file_gen  = os.path.join(args.save_gen_imgs,from_subset,fid_modality)
                    fid = fid_score.from_file_png(file_real, file_gen)
                    if from_subset=='prior':
                        fids[fid_modality].append(fid) 
                    else:
                        mods_fic[fid_modality]=fid
                
                if from_subset!='prior':
                    cond_fid[from_subset]=mods_fic
            all_fid_dicts.append(cond_fid)

        if args.calculate_loglik:
            H = model.subsets_weights()['H']
            C = tf.math.log(1./len(model.subsets.keys()))

            # placeholder
            lhoods = placeholder_llik(model)
           
            # calculate log-likelihood
            for i, data_batch in enumerate(te_data):
                data_m0 = data_batch[list(model.modalities.keys())[0]] #selecting 1st modality to infer size
                llik = model.evaluate(data_batch,calculate_llik=True,batch_size=data_m0.shape[0])

                for s_key in model.subsets.keys():
                    temp_dic = llik[s_key]
                    for m_key in temp_dic.keys():
                        lhoods[s_key][m_key].append(llik[s_key][m_key].numpy())
            
            for k, s_key in enumerate(lhoods.keys()):
                lh_subset = lhoods[s_key]
                for l, m_key in enumerate(lh_subset.keys()):
                    mean_val = tf.reduce_mean(lh_subset[m_key])
                    lhoods[s_key][m_key] = mean_val.numpy() + H.numpy() + C.numpy()

            # extract join llik
            all_llik['mnist'].append(lhoods['mnist']['joint'])
            all_llik['svhn'].append(lhoods['svhn']['joint'])
            all_llik['text'].append(lhoods['text']['joint'])
            all_llik['mnist_svhn'].append(lhoods['mnist_svhn']['joint'])
            all_llik['mnist_text'].append(lhoods['mnist_text']['joint'])
            all_llik['svhn_text'].append(lhoods['svhn_text']['joint'])
            all_llik['mnist_svhn_text'].append(lhoods['mnist_svhn_text']['joint'])

        try:
            del xhat_te, z_te
        except:
            pass

    if args.calculate_loglik:
        print('------------- Loglikelihood --------------')
        agg = []
        for i,subset in enumerate(lhoods.keys()):
            avg_llik = np.mean(all_llik[subset])
            std_llik = np.std(all_llik[subset])
            print('joint llik conditioned on {} is {:.4f} ({:.4f})'.format(subset,avg_llik,std_llik))
            agg.append(avg_llik)
        print('overall llik {:.4f} ({:.4f})'.format(np.mean(agg),np.std(agg)))

    if args.calculate_fid:
        print('--------- Unconditional FID --------------')
        all_uncon_fid = []
        for modality in fids.keys():
            avg_fid = np.mean(fids[modality])
            std_fid = np.std(fids[modality])
            all_uncon_fid.append(fids[modality])
            print('FID for {} is {:.2f} ({:.2f})'.format(modality,avg_fid,std_fid))
        all_uncon = np.array(all_uncon_fid)
        overall_avg = np.mean(all_uncon,axis=0)
        print('average unconditional FID {:.2f}'.format(np.mean(overall_avg)))
        print('with std {:.2f}'.format(np.std(overall_avg)))
   
        user_dict = pd.DataFrame(all_fid_dicts)

        all_confid = pd.DataFrame.from_dict({(i,j): user_dict[i][j] 
                                                for i in user_dict.keys() 
                                                for j in user_dict[i].keys()},
                                                orient='index')
        
        print('------------- Average Conditional FID -------------')
        print(all_confid.groupby(level=0).mean())
        print('averall conditional FID {:.2f}'.format(all_confid.mean(numeric_only=True).mean()))
        print('with std {:.2f}'.format(all_confid.groupby(level=0).std(ddof=0).mean(numeric_only=True).mean()))
    

    if args.lr_acc:
        agg = []
        print('---------------- LR acc ------------------')
        for subset in self_acc.keys():
            avg_acc = np.mean(self_acc[subset])
            std_acc = np.std(self_acc[subset])
            print('Self-accurary for subsets {} is {:.4f} ({:.4f})'.format(subset,avg_acc,std_acc))
            agg.append(avg_acc)
        print('overall lr acc {:.4f} with std between subsets {:.4f}'.format(np.mean(agg),np.std(agg)))
        

    if args.coherence_joint:
        print('------------- Joint coherence -------------')
        print(np.mean(coherence_all), np.std(coherence_all))

    if args.coherence_acc:
        user_dict = pd.DataFrame(all_coherence_dicts)

        all_coherence = pd.DataFrame.from_dict({(i,j): user_dict[i][j] 
                                                for i in user_dict.keys() 
                                                for j in user_dict[i].keys()},
                                                orient='index')
        
        with open("../output/cohe_acc_"+str(args.beta_kl)+".pkl", 'wb') as f:
            pickle.dump(all_coherence, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # NOTE columns indicated the modality being generated based on the different subsets
        # of modalities listed row-wise
        print('------------- Average coherence -------------')
        print(all_coherence.groupby(level=0).mean())
        print('--------------- Std coherence ---------------')
        print(all_coherence.groupby(level=0).std(ddof=0))
        print('averall coherence {:.4f}'.format(all_coherence.mean(numeric_only=True).mean()))
        print('average conditional coherence {:.2f} with std {:.2f}'.format(np.mean(all_avg_cohe),np.std(all_avg_cohe)))
    
    print('elapsed time: {}'.format(time.time()-start))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt_folder', default='ckpts')
    parser.add_argument('--restore_from_wandb', action='store_true')
    parser.add_argument('--restore_ckpt',default='120')
    parser.add_argument('--account_name', default=None)
    parser.add_argument('--project_name', default='codevae_mst')
    parser.add_argument('--wandb_run', nargs='+')
    parser.add_argument('--crossmodal_generation', action='store_true')
    parser.add_argument('--grid_mode', action='store_true')
    parser.add_argument('--calculate_fid', action='store_true', help='conditional and unconditional fids')
    parser.add_argument('--calculate_loglik', action='store_true')
    parser.add_argument('--calculate_trace', action='store_true')
    parser.add_argument('--modality_specific', action='store_true')
    parser.add_argument('--lr_acc', action='store_true', help='linear classification accuracy')
    parser.add_argument('--coherence_acc', action='store_true', help='condiditional coherence accuracy')
    parser.add_argument('--coherence_joint', action='store_true', help='unconditional coherence accuracy')
    parser.add_argument('--save_gen_imgs', default=None)
    parser.add_argument('--from_subsets', default=['all'], nargs='+',help='used with fids')
    parser.add_argument('--fid_modalities', default=['mnist','svhn'], nargs='+',help='used with fids')
    parser.add_argument('--from_modalities', default=['all'], nargs='+',help='subsets used to generate modalities')
    parser.add_argument('--to_modalities', default=['mnist','svhn','text'], nargs='+',help='used to generate modalities')
    parser.add_argument('--output_file', default='codevae_mst')

    args = parser.parse_args()
    main(args)
