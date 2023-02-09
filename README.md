# Quantization and Pruning for SNNs

Code for the ISCAS23/TCAS-II paper "The Hardware Impact of Quantization and Pruning for Weights in Spiking Neural Networks" by Clemens JS Schaefer, Pooria Taheri, Mark Horeni, and Siddharth Joshi.

https://arxiv.org/abs/2302.04174

To run training:

```
python3 examples/train.py --workdir=/tmp/abc --config=examples/tcja/configs/default.py
```

## Requirements

```
absl==0.0
absl_py==1.0.0
clu==0.0.6
dm_tree==0.1.6
flax==0.4.0
jax==0.2.27
matplotlib==3.5.1
ml_collections==0.1.1
numpy==1.22.1
optax==0.1.0
pandas==1.4.0
spikingjelly==0.0.0.0.12
tensorflow==2.8.0
tensorflow_datasets==4.5.2
torch==1.13.1
tree==0.2.4
```
