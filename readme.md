# Can Encrypted Images Still Train Neural Network? Investigating Image Information and Random Vortex Transformation

## Introduction

  In this paper, we establish a framework for measuring image information content to evaluate the variation in information content during image transformations. Furthermore, based on the aforementioned framework, we propose a novel image encryption algorithm called Random Vortex Transformation. This algorithm encrypts the image using random functions while preserving the neighboring information of the pixels. The encrypted images are difficult for the human eye to distinguish, yet they allow for direct training of the encrypted images using machine learning methods.

  The effect of  Random Vortex Transformation is shown as follow.

<img src=".\figure\effect.png" alt="Experimental scheme" style="zoom:100%;" />

## Results

The experiement scheme is as followed.

<img src=".\figure\Experimental scheme.png" alt="Experimental scheme" style="zoom:80%;" />

The classification accuracy on each dataset and model are listed as followed.

|   Model   | MNIST  |        |        | Fashion-MNIST |        |        | CIFAR-10 |        |        |
| :-------: | :----: | ------ | ------ | :-----------: | ------ | ------ | :------: | ------ | ------ |
|           | origin | vor    | random |    origin     | vor    | random |  origin  | vor    | random |
| ResNet-18 | 99.20% | 98.89% | 28.99% |    94.12%     | 92.85% | 40.10% |  92.80%  | 87.56% | 42.80% |
|    ViT    | 99.47% | 99.15% | 31.84% |    94.95%     | 92.61% | 56.19% |  98.50%  | 92.04% | 50.03% |

More details are in our paper.



## File Structure

```main file_structure
├── DataProcess
│   ├── code
│   │   ├── load_cifar.m
│   │   ├── random_permutation.m
│   │   ├── cifar_vor.m
│   │   ├── mnist_fashion_vor.m
│   │   └── Random_Function.m
│   ├── dataset
│   │   ├── cifar
│   │   │   ├── origin
│   │   │   ├── random
│   │   │   └── vor
│   │   ├── fashion
│   │   │   ├── origin
│   │   │   ├── random
│   │   │   │   ├── fashion_test_random_v1.mat
│   │   │   │   └── fashion_train_random_v1.mat
│   │   │   └── vor
│   │   └── mnist
│   │       ├── origin
│   │       ├── random
│   │       │   ├── mnist_test_random_v1.mat
│   │       │   └── mnist_train_random_v1.mat
│   │       └── vor
│   ├── origin_dataset
│   │   ├── cifar
│   │   ├── fashion
│   │   │   └── fashion.mat
│   │   └── mnist
│   │       └── mnist.mat
│   └── params
│       ├── cifar
│       │   └── vor_params_cifar.mat
│       ├── fashion
│       │   └── vor_params_fashion.mat
│       └── mnist
│           ├── vor_params_mnist_result.mat
│           ├── vor_params_mnist_sensitivity1.mat
│           ├── vor_params_mnist_sensitivity2.mat
│           └── vor_params_mnist_sensitivity3.mat
├── readme.md
└── figure
└── Train
    ├── dataloader.py
    ├── main.py
    └── utils.py
```

There are totally two main folders, the folder `DataProcess` is used for generating vortex-transformed dataset and random-permutated dataset, and the folder `Train` is used for training models using the datasets generated above.

In `DataProcess` folder, There are three folders.

- Folder `origin_dataset` contains the origin version of MNIST, Fashion and CIFAR-10 in `.mat`. Pay attention that you have to download the MatLab version CIFAR-10 dataset at https://www.cs.toronto.edu/~kriz/cifar.html and put it under the folder`cifar/vor/`.

- Folder `dataset` stores the original dataset, vortex-transformed dataset and random-permutated dataset used for training model. We have offered you the random-permutated dataset of MNIST and Fashion, and you should generate the random-permutated dataset of CIFAR-10 through `ramdom_permutatiom.m`.

- Folder `params` stores the random vortex parameters we used in our experiment. You can use the parameters to generate the corresponding dataset by the code offered in folder `code ` and reproduce our experiment results. You can also generate your own Random Vortex Transfoamtion parameters and generate the vortex-transformed dataset on you need. The function of the files in folder `code` are as followed.
  
  - `load_cifar.m`:load CIFAR-10 dataset
  - `random_permutation.m`:generate the ramdom-permutated dataset of CIFAR-10.
  
  - `cifar_vor.m`: generate the vortex-transformed dataset of CIFAR-10
  - `mnist_fashion_vor.m`:generate the vortex-transformed dataset of MNIST and Fashion
  - `Random_Function.m`: randomly create the vortex parameters by using random function.

In `Train` folder, there are three files.

- `dataloader.py`: used for loading data.
- `utils.py`: stores some utility functions.
- `main.py`: train models.



## How to Train
The training methods and the training hyper-parameters for training models on MNIST, Fashion and CIFAR-10 are as followed.

To run the following construction, first get into folder `Train`.
```python
cd Train
```
### CIFAR10 

#### ResNet18

- origin	


```python
python main.py --dataset cifar10 --model ResNet18 --epoch 100 --batch 256 --crop 32 
```

- vor

```python
python main.py --dataset cifar10_vor --model ResNet18 --epoch 100 --batch 32 -crop 224 --beta_of_ricap 0.3 
```
- random


```python
python main.py --dataset cifar10_random --model ResNet18 --epoch 100 --batch 256 --crop 224
```

#### ViT

- origin

```
python main.py --dataset cifar10 --model ViT --epoch 70 --batch 128 --crop 224 --lr 0.005
```

- vor

```python
python main.py --dataset cifar10_vor --model ViT --epoch 70 --batch 128 --crop 224 --lr 0.005 --wd 0.001 
```

- random

```
python main.py --dataset cifar10_random --model ViT --epoch 70 --batch 128 --crop 224 --lr 0.005
```



### MNIST

#### ResNet18

- origin

```
python main.py --dataset mnist --model ResNet18 --epoch 30 --batch 256 --crop 28
```

- vor


```
python main.py --dataset mnist_vor --model ResNet18 --epoch 30 --batch 256 --crop 224 --dev 3
```

- random

```
python main.py --dataset mnist_random --model ResNet18 --epoch 30 --batch 256 --crop 28
```

#### ViT

- orgin

```
python main.py --dataset mnist --model ViT --epoch 30 --batch 128 --crop 224
```

- vor

```python
python main.py --dataset mnist_vor --model ViT --epoch 30 --batch 128 --crop 224 --lr 0.005 --wd 0.0002
```
- random

```
python main.py --dataset mnist_random --model ViT --epoch 30 --batch 128 --crop 224
```



### Fashion

#### ResNet18

- origin

```
python main.py --dataset fashion --model ResNet18 --epoch 50 --batch 256
```

- vortex 

```
python main.py --dataset fashion_vor --model ResNet18 --epoch 50 --batch 256 --crop 224 --wd 0.0005 
```

- random

```
python main.py --dataset fashion_random --model ResNet18 --epoch 50 --batch 256 --crop 224 
```

#### ViT

- origin

```
python main.py --dataset fashion --model ViT --epoch 50 --batch 128 --crop 224 --lr 0.005 --wd 0.0002 
```

- vortex 

```
python main.py --dataset fashion_vor --model ViT --epoch 50 --batch 128 --crop 224 --lr 0.005 --wd 0.0005
```

- random

```
python main.py --dataset fashion_random --model ViT --epoch 50 --batch 128 --crop 224 --lr 0.005 --wd 0.0002 
```



We trained all the models on one GPU (GeForce RTX 3090). 

Pay attention that if you use your own Random Vortex Transfoamtion parameters, the hyper-parameters mentioned above should be modified slightly.
