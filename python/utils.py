import numpy as np
import itertools
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from scipy.io import loadmat
import random
import shutil

digit_text_english = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'];

def save_generated_samples(samples,real_imgs,dir_save='../output/fid', make_zip=False,name='fid'):
    if  os.path.exists(dir_save): 
        print('deleting {} folder tree...'.format(dir_save))
        shutil.rmtree(dir_save, ignore_errors=True)

    samples_keys = list(samples.keys()) + ['real']
    samples_subset = list(samples['prior'].keys())
    for subset in samples_keys:
        for modality in samples_subset:
            dir_f = os.path.join(dir_save, subset, modality)
            if not os.path.exists(dir_f):
                os.makedirs(dir_f)

    for subset in samples_keys:
        if subset != 'real':
            samples_subset = samples[subset]
            print('saving modalities generated from subset {}'.format(subset))
            for modality, gens in samples_subset.items():
                if modality!='text':
                    dir_f = os.path.join(dir_save, subset, modality)
                    for j in range(gens.shape[0]):
                        tf.keras.utils.save_img(os.path.join(dir_f,'img_'+str(j)+'.png'), gens[j,...], data_format='channels_last')

        else:
            for modality, gens in real_imgs.items():
                if modality not in ['text','target']:
                    print('saving real {} images'.format(modality))
                    dir_f = os.path.join(dir_save, subset, modality)
                    for j in range(gens.shape[0]):
                        tf.keras.utils.save_img(os.path.join(dir_f,'img_'+str(j)+'.png'), gens[j,...], data_format='channels_last')
    if make_zip:
        print('ziping FID folder...')
        shutil.make_archive('../output/'+name, format='zip', root_dir=dir_save)

def load_mnist_svhn_text(alphabet, path='../data/raw', nr_tr=1121360, nr_te=200000):
    # matching index based on Shi et al. 2019
    mnist_tr_idx = np.load(os.path.join(path,'mnist_idx_tr.npy'))
    mnist_te_idx = np.load(os.path.join(path,'mnist_idx_te.npy'))
    svhn_tr_idx = np.load(os.path.join(path,'svhn_idx_tr.npy'))
    svhn_te_idx = np.load(os.path.join(path,'svhn_idx_te.npy'))
    
    shvm_te = loadmat(os.path.join(path,'test_32x32.mat'))
    x2_te = shvm_te['X']
    #x2_te = batch_first(x2_te)
    x2_te = np.transpose(x2_te, (3, 0, 1, 2))
    shvm_tr = loadmat(os.path.join(path,'train_32x32.mat'))
    x2_tr = shvm_tr['X']
    #x2_tr = batch_first(x2_tr)
    x2_tr = np.transpose(x2_tr, (3, 0, 1, 2))

    x1_tr = np.load(os.path.join(path,'mnist_tr.npy'))
    x1_te = np.load(os.path.join(path,'mnist_te.npy'))
    y_tr = np.load(os.path.join(path,'mnist_y_tr.npy'))
    y_te = np.load(os.path.join(path,'mnist_y_te.npy'))
    
    # reshape mnist
    x1_te = x1_te.reshape((x1_te.shape[0], x1_te.shape[1], x1_te.shape[2],1)) 
    x1_tr = x1_tr.reshape((x1_tr.shape[0], x1_tr.shape[1], x1_tr.shape[2],1)) 
    
    #scale 
    x2_tr = x2_tr/255.
    x2_te = x2_te/255.
    x1_tr = x1_tr/255.
    x1_te = x1_te/255.
    
    # randomly selected idx
    # matching index size is larger than nr_tr and nr_te
    idx_tr_val = random.sample(range(svhn_tr_idx.shape[0]),nr_tr)
    idx_te_val = random.sample(range(svhn_te_idx.shape[0]),nr_te)

    target_tr = y_tr[mnist_tr_idx[idx_tr_val]]
    text_tr = []
    print('generating text label...')
    for i in target_tr:
        text_tr.append(create_text_from_label_mnist(i, alphabet))
    text_tr = np.array(text_tr)

    target_te = y_te[mnist_te_idx[idx_te_val]]
    text_te = []
    for i in target_te:
        text_te.append(create_text_from_label_mnist(i, alphabet))
    text_te = np.array(text_te)

    all_data = (x1_tr[mnist_tr_idx[idx_tr_val],:].astype(np.float32),
                x2_tr[svhn_tr_idx[idx_tr_val],:].astype(np.float32),
                text_tr,
                tf.keras.utils.to_categorical(y_tr[mnist_tr_idx[idx_tr_val]], num_classes=10),
                x1_te[mnist_te_idx[idx_te_val],:].astype(np.float32),
                x2_te[svhn_te_idx[idx_te_val],:].astype(np.float32),
                text_te,
                tf.keras.utils.to_categorical(y_te[mnist_te_idx[idx_te_val]], num_classes=10),
                )

    return all_data

def create_text_from_label_mnist(label, alphabet, len_seq=8):
    text = digit_text_english[label];
    sequence = len_seq * [' '];
    start_index = random.randint(0, len_seq - 1 - len(text));
    sequence[start_index:start_index + len(text)] = text;
    sequence_one_hot = one_hot_encode(len_seq, alphabet, sequence);
    return sequence_one_hot

def one_hot_encode(len_seq, alphabet, seq):
    X =  np.zeros((len_seq, len(alphabet)))

    if len(seq) > len_seq:
        seq = seq[:len_seq];
    for index_char, char in enumerate(seq):
        if char2Index(alphabet, char) != -1:
            X[index_char, char2Index(alphabet, char)] = 1
    return X

def char2Index(alphabet, character):
    return alphabet.find(character)

def concat_nested_dict(dict_old,dict_new,subsets,modalities,iteration):
    #concat all gen modalities
    if iteration == 0:
        for s in subsets:
            for m in modalities:
                dict_old[s][m] = dict_new[s][m]
    else:
        for s in subsets:
            for m in modalities:
                dict_old[s][m] = tf.concat([dict_old[s][m],dict_new[s][m]],0)
    return dict_old 

def concat_dict(dict_old,dict_new,subsets,iteration):
    #concat all gen modalities
    if iteration == 0:
        for s in subsets:
            dict_old[s] = dict_new[s]
    else:
        for s in subsets:
            dict_old[s] = tf.concat([dict_old[s],dict_new[s]],0)
    return dict_old

def sample_xhat_zte(model, te_dict, args, z_method='sample'):
    batch_size = args.batch_size
    te_data = tf.data.Dataset.from_tensor_slices(te_dict).batch(batch_size)

    # placeholder to concat all gen modalities
    all_subsets = list(model.subsets.keys()) + ['prior']
    xte_hat = {}
    for s in all_subsets:
        xte_hat[s] = {}
    z_te = {}

    #concat_nested_dict(dict_old,dict_new,subsets,modalities,iteration):
    for i, data_batch in enumerate(te_data):
        z_te_b, x_hat_b = model.evaluate(data_batch, z_method=z_method)
        z_te    = concat_dict(z_te, z_te_b, model.subsets.keys(), i)
        xte_hat = concat_nested_dict(xte_hat, x_hat_b, all_subsets, model.modalities.keys(), i)

    return xte_hat, z_te
