# AFSC - Active Few Shot Classification : a New Paradigm for Data-Scarce Learning Settings
This repository is the official implementation of [AFSC - Active Few Shot Classification : a New Paradigm for Data-Scarce Learning Settings]


AFSC is a new formulation of classical Transductive Few-Shot Classification that has the potential of reaching very high scores. We rely on active selection of labels based on data distributions. We use two-tier methodology that consists in inferring the distribution of samples among the classes, and selecting the samples that are the most probable in each class to be labeled first, then the least probable ones.

## Downloading the datasets

For this work, we used two different feature extractors:

### EASY ensembled resnet 18.
mini-ImageNet, tiered-ImageNet, CUB, FC100 datasets post feature extractation are available for download : [Google Drive link](https://drive.google.com/drive/folders/1fMeapvuR6Rby0HDHd5L74BEXRyiOF942) in features/resnet12.
- Download, for all three backbones, the four datasets in the format : <dataset_name>featuresAS<backbone_number>.pt11" and place them in /datasets folder.
More on this on EASY's github repository: https://github.com/ybendou/easy

### TIM resnet 12.
The datasets features (mini-ImageNet, tiered-ImageNet, CUB) are not available directly for download but we have access to the models in [Google Drive link](https://drive.google.com/drive/folders/1SeIYcqST71b00zR-9LSi47QKAZCxayWv).
- First, download the datasets in /data and unzip them in local folder /datasets/data.
- Second, download "checkpoints" from models/ in the drive and unzip it local folder /models.
- Then, run the datasets.py file with the required dataset's name as argument:
    for mini-ImageNet : $ python .\datasets.py --dataset mini
    for tiered-ImageNet : $ python .\datasets.py --dataset tiered
    for CUB : $ python .\datasets.py --dataset cub
This will save the extracted features of the selected dataset in /datasets.

More on this on TIM's github repository: https://github.com/mboudiaf/TIM

## Inference

### Important arguments

- `dataset`: choices=['mini', 'cub','tiered', 'fc100']
- `model`: choices=['resnet12', 'resnet18'] (resnet18 not available for fc100 dataset)
- `n_ways`: number of classes for the few shot task.
- `n_samples` : number of samples used in the few shot task (not counting the labels).
- `n_labels` : number of labels used for the few-shot task.
- `gs` : boolean for using graph smoothing on data when set to true.
- `crit` :  active selection criterion used for the few shot tasks choices=['LSS', 'K-medoid','margin', 'oracle', 'random']

### Comparison with current benchmarks

#### mini-ImageNet 

| **mini-Imagenet**                   | ****          | ****         | ****         | ****        | ****        |
|-------------------------------------|---------------|--------------|--------------|-------------|-------------|
| **Shots configuration**             | method        | 5-labels     | 25-labels    | 50-labels   | 100-labels  |
| **Equal labels per class (random)** | PT-MAP        | 62.6         | 75.2         | 81.3        | 70.4        |
|                           | LaplacianShot | 67.6         | 86.2         | 90.5        | 94          |
|                         | BD-CSPN       | 69.0         | 85.1         | 89.7        | 93.4        |
|                                | TIM           | 69.3         | 84.5         | 89.4        | 93.2        |
|                            | α-TIM         | 69.4         | 86.9         | 91.6        | 94.8        |
| **random label sampling**           | soft K-means  | 70.7 ± 0.28  | 83.4 ± 0.16  | 87.7 ± 0.12 | 91.8 ± 0.09 |
| **Active label sampling**           | soft K-means  | 64.3 ± 0.29  | 84.7 ± 0.14  | 89.6 ± 0.09 | 93.3 ± 0.06 |
|                               | LSS (ours)    | 73.5 ± 0.21  | 87.7 ± 0.16  | 93.8 ± 0.11 | 96.8 ± 0.07 |
| **tiered-Imagenet**                 |               |              |              |             |             |
| **Shots configuration**             | method        | 5-labels     | 25-labels    | 50-labels   | 100-labels  |
| **Equal labels per class (random)** | PT-MAP        | 63.6         | 77.5         | 83.1        | 88.6        |
|                            | LaplacianShot | 74.0         | 89.3         | 92.8        | 95.3        |
|                           | BD-CSPN       | 75.8         | 88.7         | 92.0        | 94.8        |
|                            | TIM           | 75.8         | 88.1         | 81.2        | 94.6        |
|                         | α-TIM         | 76           | 89.9         | 93.6        | 96.1        |
| **Random label sampling**           | soft K-means  | 74.7 ± 0.29  | 84.8 ± 0.18  | 88.8 ± 0.14 | 92.6 ± 0.10 |
| **Active label sampling**           | soft K-means  | 66.9 ± 0.30  | 85.7 ± 0.16  | 90.0 ± 0.11 | 93.5 ± 0.07 |
|                           | LSS (ours)    | 76.1 ± 0.24  | 87.5 ± 0.17  | 92.2 ± 0.13 | 95.8 ± 0.09 |


#### CUB 

| **Shots configuration** | **method**    | **5-labels** |
|-------------------------|---------------|--------------|
| **Equal labels per class (random)**           | PT-MAP        | 67.3         |
|                         | LaplacianShot | 84.7         |
|                         | BD-CSPN       | 76.1         |
|                         | TIM           | 76.4         |
|                         | α-TIM         | 77.2         |
|                         | soft K-means  | 75.9 ± 0.28  |
| **Random label sampling**         | soft K-means  | 66.6 ± 0.29  |
| **Active label sampling**        | LSS (ours)    | 77.2 ± 0.21  |

To reproduce the results of these tables (last two lines) run:

    $ python main.py --dataset <dataset-name> --model resnet18 --n_labels <number-of-labels>
    
dataset-name =  CUB, miniImageNet, tieredImagenet\
n_labels = 5, 25, 50, 100


### Comparison of active selection criteria

| **labels ** | **sampling strategy** | **mini-ImageNet** | **tiered-ImageNet** | **CUB**         | **FC-100**      |
|:-----------:|:---------------------:|:-----------------:|:-------------------:|:---------------:|:---------------:|
| 5 labels    | random                | 62.2 ± 0.31   | 61.2 ± 0.30     | 67.7 ± 0.33 | 46.5 ± 0.26 |
|             | oracle                | 95.2 ± 0.21   | 97.4 ± 0.3      | 95.5 ± 0.18 | 80.2 ± 1.40 |
|             | margin                | 69.0 ± 0.30   | 69.0 ± 0.26     | 78.5 ± 0.30 | 48.7 ± 0.26 |
|             | K-medoid              | 75.6 ± 0.28   | 75.2 ± 0.27     | 83.8 ± 0.25 | 52.1 ± 0.25 |
|             | LSS (ours)            | 76.1 ± 0.26   | 75.7 ± 0.27     | 84.2 ± 0.24 | 53.3 ± 0.24 |
| 25 labels   | random                | 86.8 ± 0.18   | 85.6 ± 0.20     |        -        | 68.8 ± 0.20 |
|             | margin                | 88.6 ± 0.19   | 87.0 ± 0.18     |        -        | 69.7 ± 0.21 |
|             | K-medoid              | 89.7 ± 0.17   | 88.6 ± 0.16     |        -        | 71.2 ± 0.19 |
|             | LSS (ours)            | 90.9 ± 0.16   | 89.3 ± 0.19     |        -        | 71.7 ± 0.20 |

To reproduce the results of this tables 

    $ python main.py --n_labels <number-of-labels> --dataset <dataset-name> --model resnet12 --crit <criterion>
    
n_labels = 5, 25\
dataset-name =  CUB, miniImageNet, tieredImagenet, FC-100 \ 
criterion = random, oracle, margin, K-medoid, LSS
    




