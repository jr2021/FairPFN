from priors.fairmlp import *
import pickle as pkl
import torch

with open('artifacts/config_sample.pkl', 'rb') as f:
    config_sample_ = pkl.load(f)


config_sample_['prior_mlp_activations'] = torch.nn.modules.activation.Tanh()
config_sample_['num_layers'] = 5
config_sample_['prior_mlp_hidden_dim'] = 10
config_sample_['num_causes'] = 5
config_sample_['noise_std'] = 0.03052169541506959
config_sample_['prior_mlp_dropout_prob'] = 3.2991787263429043e-0
num_samples = 100

prior_dataset = get_batch(batch_size=1, num_features=3, num_prot_attrs=1, seq_len=num_samples, device='cpu', hyperparameters=config_sample_)