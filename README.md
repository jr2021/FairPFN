# FairPFN ⚖️🚀

Attached is the code supporting the ICML submission "FairPFN: A Tabular Foundation Model for Causal Fairness."

### Installation

Create a conda environment with python version 3.10, activate it, and install the requirements in requirements.txt

```
   conda create -n fairpfn_env python=3.10
   conda activate fairpfn_env
   pip install -r requirements.txt
```


### Running FairPFN

Run ```inference_example.py``` to play around with FairPFN on any of our real-world and synthetic benchmark datasets by changing the ```dataset``` variable (runs on Law School by default)

### Sampling data from the prior

Run ```prior_data_example.py``` to sample data from our prior. You can change the values of listed hyperparameters in ```config_sample_``` to change the MLP/SCM that the data is sampled is from.
