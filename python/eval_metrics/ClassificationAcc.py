from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score
import tensorflow as tf
import numpy as np
import random

class ClassificationAccuracy:
    def __init__(self, metric = 'accuracy'):
        self.metric = metric
        if metric == 'accuracy':
            self.score = accuracy_score
        elif metric == 'avg_precision':
            self.score = average_precision_score

    def train_cls(self, z, y):
        #train lr
        z = z.numpy()
        
        cls_lr = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=3000)
        cls_lr.fit(z, y)
        
        return cls_lr 

    def train_all_cls(self, z_dict, tr_target,labels=None):
        all_cls = dict()
        if labels is None:
            for k in z_dict.keys():
                z = z_dict[k]
                
                y = np.argmax(tr_target,1)
                cls = self.train_cls(z,y)
                all_cls[k] = cls
        else:
            for l,l_str in enumerate(labels):
                rep_cls = dict()
                for k in z_dict.keys():
                    z = z_dict[k]
                    
                    y = tr_target[:,l]
                    cls = self.train_cls(z,y)
                    rep_cls[k] = cls
                all_cls[l_str]=rep_cls

        self.all_cls = all_cls

    def test_lr(self, cls, z_te, te_target):
        z_te = z_te.numpy()
        
        if self.metric == 'accuracy':
            y_te = np.argmax(te_target,1)
            y_pred = cls.predict(z_te)
            score = self.score(y_te, y_pred)
        elif self.metric == 'avg_precision':
            y_pred = cls.predict(z_te)
            score = self.score(te_target, y_pred)
        return score

    def get_accuracy(self, tr_target, te_target, z_dict_tr, z_dict_te, print_acc=False, labels=None):
        print('training cls')
        self.train_all_cls(z_dict_tr, tr_target, labels=labels)
        
        acc_subset = dict()
        acc_self   = dict()
        acc_cross  = dict()
        # if labels are passed, keys() are 40 attr for celeba
        if labels is None:
            for key_cls in self.all_cls.keys():
                cls = self.all_cls[key_cls]
                acc_allsubsets = dict()
                acc_allcross   = dict()
                for key_subset in z_dict_te.keys():
                    z_te = z_dict_te[key_subset]
                    acc = self.test_lr(cls, z_te, te_target)
                    # this saves all
                    acc_allsubsets[key_subset] = acc

                    if key_cls!=key_subset:
                        acc_allcross[key_subset] = acc
                    elif key_cls==key_subset:
                        acc_self[key_cls] = acc
                        
                    if print_acc:
                        print('cls trained with {} and tested on z from {} has acc: {:0.4f}'
                                .format(key_cls, key_subset, acc))
                acc_subset[key_cls] = acc_allsubsets
                acc_cross[key_cls] = acc_allcross
        else:
            for l, l_str in enumerate(labels):
                attr_dic = self.all_cls[l_str]
                y_te = te_target[:,l]
                acc_attr=dict()
                for key_cls in attr_dic.keys():
                    cls = attr_dic[key_cls]
                    for key_subset in z_dict_te.keys():
                        if key_cls==key_subset:
                            z_te = z_dict_te[key_subset]
                            acc = self.test_lr(cls, z_te, y_te)
                            acc_attr[key_cls] = acc
                acc_self[l_str]=acc_attr
                            
        return acc_subset, acc_cross, acc_self
