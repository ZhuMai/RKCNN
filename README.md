# RKCNN


This repository is for [Convolutional neural networks combined with Runge–Kutta methods](https://doi.org/10.1007/s00521-022-07785-2).

### citation
If you find RKCNN useful in your research, please consider citing:

	@article{zhu2022convolutional,
	  title={Convolutional neural networks combined with Runge--Kutta methods},
	  author={Zhu, Mai and Chang, Bo and Fu, Chong},
	  journal={Neural Computing and Applications},
	  volume={35},
	  pages={1629–1643},
	  year={2023},
	  publisher={Springer},
	  doi="10.1007/s00521-022-07785-2"
	}

An example to train an RKCNN-E-5_5_5 with growth rate 80 on CIFAR-10:

```bash
python3 train_cifar.py --out_features 10 --update1 0 --update2 0 --update3 0 --k1 80 --k2 80 --k3 80 --s1 5 --s2 5 --s3 5 --batch-size 32 --attention --bottleneck --data_augmentation --keep_prob 1
```

An example to train an RKCNN-I-5_5_5 with growth rate 80 on CIFAR-100:

```bash
python3 train_cifar.py --out_features 100 --replace --k1 80 --k2 80 --k3 80 --s1 5 --s2 5 --s3 5 --batch-size 32 --attention --bottleneck --data_augmentation --keep_prob 1
```

An example to train an RKCNN-R-5_5_5 with growth rate 80 on CIFAR-100:

```bash
python3 train_cifar.py --out_features 100 --k1 80 --k2 80 --k3 80 --s1 5 --s2 5 --s3 5 --batch-size 32 --attention --bottleneck --data_augmentation --keep_prob 1
```
This article and repository are used for image classification. If you are interested in semantic segmentation, you can refer to [RKSeg](https://github.com/ZhuMai/RKSeg) and [RKSeg+](https://github.com/ZhuMai/RKSegPlus).
