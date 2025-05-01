from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score
import tensorflow as tf
import numpy as np
import random
import wandb
import os
import re
from warnings import simplefilter
simplefilter(action='ignore', category=UserWarning)

def findWord(w):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

class CoherenceAccuracy:
    def __init__(self, modality_cls, metric='accuracy'):
        self.modalities = modality_cls
        
        self.metric = metric
        if metric == 'accuracy':
            self.score = accuracy_score
        elif metric == 'avg_precision':
            self.score = average_precision_score

    def load_cls(self,dir_path='./cls_models'):
        all_cls = dict() 
        for key, cls_model in self.modalities.items():

            cls = cls_model()

            # provide checkpoint and manager
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-4)
            checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=cls)
            manager = tf.train.CheckpointManager(checkpoint,os.path.join(dir_path,key),max_to_keep=3)
            
            # restore last model if exits
            checkpoint.restore(manager.latest_checkpoint).expect_partial()
            all_cls[key] = cls
            if manager.latest_checkpoint:
                print("Classifier restored from {}".format(manager.latest_checkpoint))
            else:
                print("Classifier does not exist in the specified folder")

            self.all_cls = all_cls
        print('finished loading cls')

    def test_cls(self, data_set, cls_model, labels=None):
        if labels is None:
            all_acc = []
            for i, (x,y) in enumerate(data_set):
                y_hat = cls_model.test((x,y))
                acc  = self.score(y, y_hat)
                all_acc.append(acc)
            avg_all_acc = np.mean(all_acc)
        else:
            avg_all_acc=dict()
            for l, l_str in enumerate(labels):
                all_acc = []
                for i, (x,y) in enumerate(data_set):
                    y_hat = cls_model.test((x,y))
                    pred  = y_hat[:,l]
                    y_te  = y[:,l]
                    acc   = self.score(y_te, pred)
                    all_acc.append(acc)
                avg_all_acc[l_str] = np.mean(all_acc)
        
        return avg_all_acc
   
    
    def joint_coherence(self, modalityhat_dict, te_target, batch_size=100, dir_path='./cls_models',labels=None):
        print('loading pre-trained cls...')
        self.load_cls(dir_path)
    
        print('calculating joint coherance...')
        for subset_key in modalityhat_dict.keys():
            if subset_key == 'prior':
                modalities = modalityhat_dict[subset_key]
                pred_mods  = np.zeros((len(modalities.keys()), te_target.shape[0]))

                cnt=0
                for modality_key, modality_hat in modalities.items():
                    data=tf.data.Dataset.from_tensor_slices((modality_hat, te_target)).batch(batch_size)
                    cls = self.all_cls[modality_key]
           
                    pred_mod=[]
                    for i, (x,y) in enumerate(data):
                        y_hat = cls.test((x,y))
                        pred =  np.argmax(y_hat.numpy(), axis=1).astype(int)
                        pred_mod.append(pred)
                    pred_mod = np.concatenate(pred_mod,0)
                    pred_mods[cnt,:] = pred_mod
                    cnt+=1
                
        coh_mods = np.all(pred_mods == pred_mods[0, :], axis=0)
        joint_coherence = np.sum(coh_mods.astype(int))/float(te_target.shape[0])
        
        return joint_coherence

    def get_coherence(self, modalityhat_dict, te_target, batch_size=100, print_acc=False, dir_path='./cls_models',labels=None):
        # load pre-trained cls
        print('loading pre-trained cls...')
        self.load_cls(dir_path)
        
        acc_subset = dict()
        acc_cross  = dict()
        # prior is the last key
        full_set = list(modalityhat_dict.keys())[-2]
        for subset_key in modalityhat_dict.keys():
            modalities = modalityhat_dict[subset_key]
            acc_modalities = dict()
            acc_cross_mods  = dict()
            for modality_key, modality_hat in modalities.items():
                data=tf.data.Dataset.from_tensor_slices((modality_hat, te_target)).batch(batch_size)
                cls = self.all_cls[modality_key]
                acc = self.test_cls(data,cls,labels=labels) 
                acc_modalities[modality_key] = acc
                if subset_key not in ('prior',full_set):
                    if not findWord(modality_key)(subset_key.replace('_', ' ')):
                        acc_cross_mods[modality_key] = acc

            acc_subset[subset_key] = acc_modalities
            if subset_key not in ('prior',full_set):
                acc_cross[subset_key]  = acc_cross_mods
        
        return acc_subset, acc_cross
