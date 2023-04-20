# Comparison to Norse

Small example to Norse MNIST example.

Norse example (I had to experiment with how to load the package):
```
python3 norse.py
```

JAX example (this work):
```
python3 train.py --workdir=/tmp/abc --config=config.py
```

Ran it on Quadro RTX 6000 norse 42.07s/it vs this work ~7.5s/it.