import argparse
import os
import random
import numpy as np

parser = argparse.ArgumentParser(description="""Optimized code for training usual datasets/model
Examples of use (to reach peak accuracy, not for fastest prototyping):
To train MNIST with 99.64% accuracy (5 minutes):
python main.py --epochs 30 --milestones 10 --dataset MNIST --feature-maps 8
To train MNIST with 10% database and 99.31% accuracy (10 minutes):
python main.py --epochs 300 --dataset MNIST --dataset-size 6000 --model wideresnet --feature-maps 4 --skip-epochs 300
To train Fashion-MNIST with 96% accuracy (2 hours):
python main.py --dataset fashion --mixup
To train CIFAR10 with 95.90% accuracy (1 hour):
python main.py --mixup
To train CIFAR100 with 78.55% accuracy (93.54% top-5) (1hour):
python main.py --mixup --dataset cifar100
To train CIFAR100 with 80.12% accuracy (94.70% top-5) (4h):
python main.py --mixup --model wideresnet --feature-maps 16 --dataset CIFAR100
To train Omniglot (few-shot) with 99.85% accuracy (99.39% in 1-shot) (10minutes):
python main.py --dataset omniglotfs --dataset-device cpu --feature-maps 16 --milestones 10 --epochs 30 --preprocessing "PEME"
To train CUBFS (few-shot) with 85.24% accuracy (68.14% in 1-shot) (2h):
python main.py --dataset cubfs --mixup --rotations --preprocessing "PEME"
To train CIFARFS (few-shot) with 84.87% accuracy (70.43% in 1-shot) (1h):
python main.py --dataset cifarfs --mixup --rotations --skip-epochs 300 --preprocessing "PEME"
To train CIFARFS (few-shot) with 86.83% accuracy (70.27% in 1-shot) (3h):
python main.py --dataset cifarfs --mixup --model wideresnet --feature-maps 16 --skip-epochs 300 --rotations --preprocessing "PEME"
To train MiniImageNet (few-shot) with 80.43% accuracy (64.11% in 1-shot) (2h):
python main.py --dataset miniimagenet --model resnet12 --gamma 0.2 --milestones 30 --epochs 120 --batch-size 128 --preprocessing 'EME'
To train MiniImageNet (few-shot) with rotations and 81.63% accuracy (65.64% in 1-shot) (2h):
python main.py --dataset miniimagenet --model resnet12 --milestones 60 --epochs 240 --cosine --gamma 1 --rotations --skip-epochs 200
To train MiniImageNet (few-shot) with 83.18% accuracy (66.78% in 1-shot) (40h):
python main.py --device cuda:012 --dataset miniimagenet --model S2M2R --lr -0.001 --milestones 0 --epochs 600 --feature-maps 16 --rotations --manifold-mixup 400 --skip-epochs 600 --preprocessing "PEME"
""", formatter_class=argparse.RawTextHelpFormatter)

### hyperparameters
parser.add_argument("--n_runs", type=int, default=1000, help="number of few shot tasks to generate")
parser.add_argument("--n_ways", type=int, default=5, help="number of classes")
parser.add_argument("--n_samples", type=int, default=75, help="number of samples without counting labels")
parser.add_argument("--n_labels", type=int, default=5, help="number of labels")
parser.add_argument("--device", type=str, default='cpu', help="device cuda or cpu")
parser.add_argument("--dataset", type=str, default='mini', help="dataset used")
parser.add_argument("--model", type=str, default='resnet12', help="feature extractor architecture: EASY ensembled resnet12 or TIM resnet18")
parser.add_argument("--dataset_fn", type=list, default='["datasets/minifeaturesAS1.pt11", "datasets/minifeaturesAS2.pt11", "datasets/minifeaturesAS3.pt11"]'
, help="dataset filenames to load")
parser.add_argument("--train_c", type=int, default='64', help="number of train classes in dataset")
parser.add_argument("--val_c", type=int, default='16', help="number of validation classes in dataset")
parser.add_argument("--test_c", type=int, default='20', help="number of test classes in dataset")
parser.add_argument("--cub", type=bool, default=False, help="true when using the cub dataset")
parser.add_argument("--alpha_dir", type=int, default=2, help="dirichlet alpha value")

parser.add_argument("--run_batch_size", type=int, default=100, help="batch size for the runs")
parser.add_argument("--comb_batch_size", type=int, default=100, help="combination batching for oracle criterion")
parser.add_argument("--switch", type=int, default=1, help="round number where we switch from high to low confidence sampling")
parser.add_argument("--crit", type=str, default='LSS', help="active sampling criterion (LSS, K-medoid, margin, random or orcale)")
parser.add_argument("--inference", type=str, default='softkmeans', help="few-shot predictions model, default=softkmeans")


parser.add_argument("--gs", type=bool, default=True, help="apply graph smoothing when generating runs")
parser.add_argument("--beta", type=int, default='2', help="graph smoothing hyperparameter")
parser.add_argument("--kappa", type=int, default='3', help="graph smoothing hyperparamete")
parser.add_argument("--m", type=int, default='20', help="graph smoothing hyperparameter")

parser.add_argument("--n_init", type=int, default='10', help="soft k-means hyperparameter")
parser.add_argument("--temp", type=int, default='5', help="soft k-means hyperparamete")
parser.add_argument("--n_iter", type=int, default='30', help="soft k-means hyperparameter")

args = parser.parse_args()

if args.model == 'resnet18':
    args.val_c = 0
    args.dataset_fn = ["datasets/mini18.pt"]
elif args.model == 'resnet12':
    args.dataset_fn = ["datasets/minifeaturesAS1.pt11", "datasets/minifeaturesAS2.pt11", "datasets/minifeaturesAS3.pt11"]


if args.dataset == 'cub':
    args.cub == True
    args.train_c = 100
    args.val_c = 50
    args.test_c = 50
    if args.model == 'resnet18':
        args.dataset_fn = ["datasets/cub18.pt"]
        args.val_c = 0
    elif args.model == 'resnet12':
        args.dataset_fn = ["datasets/cubfsfeaturesAS1.pt11", "datasets/cubfsfeaturesAS2.pt11", "datasets/cubfsfeaturesAS3.pt11"]

if args.dataset == 'tiered':
    args.train_c = 351
    args.val_c = 97
    args.test_c = 160
    if args.model == 'resnet18':
        args.dataset_fn = ["datasets/tiered18.pt"]
        args.val_c = 0
    elif args.model == 'resnet12':
        args.dataset_fn = ["datasets/tieredfeaturesAS1.pt11", "datasets/tieredfeaturesAS2.pt11", "datasets/tieredfeaturesAS3.pt11"]

if args.dataset == 'fc':
    args.train_c = 60
    args.val_c = 20
    args.test_c = 20
    if args.model == 'resnet18':
        print('no resnet18 architecture available for fc100, loading ensembled resnet12 instead...')
    args.dataset_fn = ["datasets/fc100featuresAS1.pt11", "datasets/fc100featuresAS2.pt11", "datasets/fc100featuresAS3.pt11"]
