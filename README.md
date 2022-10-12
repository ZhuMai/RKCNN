# RKCNN


This repository is for [Convolutional Neural Networks Combined with Runge-Kutta Methods](https://arxiv.org/abs/1802.08831).

An example to train an RKCNN-I-5_5_5 with growth rate 80 on CIFAR-100:

```bash
CUDA_VISIBLE_DEVICES=0 python3 train_cifar.py --out_features 100 --replace --k1 80 --k2 80 --k3 80 --s1 5 --s2 5 --s3 5 --batch-size 32 --attention --bottleneck --save YOUR_PATH
```
