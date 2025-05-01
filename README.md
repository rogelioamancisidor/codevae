# Aggregation of Dependent Expert Distributions in Multimodal Variational Autoencoders
Code for the framework in **Aggregation of Dependent Expert Distributions in Multimodal Variational Autoencoders** 

## Requirements
The code for the Consensus of Dependent Experts with Variational Autoencoders (CoDE-VAE) model is developed in TensorFlow. We suggest to use `Docker` to run the code. Run the command `docker pull rogelioandrade/coevae:v3` to get an image with all dependencies needed or specify `image: "rogelioandrade/coevae:v3"` in `Kubernetes`.

The structure of the project should look like this:

```
codevae
      ├── data
      ├── output
      └── python
```

**Note**: the file `train_codevae_mst.py` uses `wandb` to log model training. However, if you don't have an account, specify`wandb_mode = 'disabled'`, which is the default value, to train the model.

## Data
### MNIST-SVHN-Text
To download the data run the following commands

```
wget -q 'https://www.dropbox.com/scl/fi/ly0etebjmtter8gxveyfa/data_codevae_mst.zip?rlkey=uokzyc8j2qihjo8o2jv3u0x0t&st=wpmu80hd&dl=0' -O data_codevae_mst.zip
unzip -q data_codevae_mst.zip -d ../data
```
which assume that the current location is `codevae/python`

## Pretrained model
To download a pretrained model run the following commands

```
wget -q 'https://www.dropbox.com/scl/fi/u2qeabmbresv7e5f7hseb/output_codevae_mst.zip?rlkey=qrsltu3m5jxddjqlz8bxlkl5r&st=5997f43q&dl=0' -O output_codevae_mst.zip
unzip -q output_codevae_mst.zip -d ./output
```
which assume that the current location is `codevae/python` and will save the `cktps` and `commandline_args.txt` in `codevae/python/output` folder. `codevae/python/output` is the default location when `wandb_mode = 'disabled'` and it is used in `test_codevae_mst.py`. 

## Usage
### Train CoDE-VAE
Use

```
python -u train_codevae_mst.py --subsets_trainable_weights --create_dset
```
to train CoDE-VAE using the default values in the `train_codevae_mst.py`. If you have a `wandb` account, use `--wandb_mode online` to log the run in your account, otherwise the trained model is saved in `codevae/python/output` by default. 

### Test CoDE-VAE
If you logged a run in your `wandb` account, you can run 

```
python -u test_codevae_mst.py --wandb_run <id_to_restore> --restore_from_wandb --account_name <your_wandb_account_name>
```
and one of the following downstream tasks: `--calculate_fid`, `--calculate_loglik`, `--calculate_trace`, `--lr_acc`, `--coherence_acc`, or `--coherence_joint`. 

If you didn't log the run in `wandb`, simply run `python -u test_codevae_mst.py` (which assumes that the trained model is saved in `codevae/python/output`) and one of the downstream tasks listed above. 

To generate `mnist` modalities conditioned on latent representations from the subset `mnist_svhn_text`, run 

```
python -u test_codevae_mst.py --crossmodal_generation --from_modalities mnist_svhn_text --to_modalities mnist 
``` 

you can add the argument `--grid_mode` to create a grid with 100 randomly generated images. Alternatively, you can pass your `wandb` details to load the model. 
