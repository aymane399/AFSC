
import logging
import os
import sys
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import Dataset

pil = transforms.ToPILImage()
tens = transforms.ToTensor()
import shutil
import argparse
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
from tqdm import tqdm
from contrastive_ds import ContrastiveLearningDataset,class_images_dict
from sklearn.neural_network import MLPClassifier as MLP
from utils import *
from sampling_utils import *

torch.cuda.empty_cache()

def load_dataset(dataset_filename, train_c, device):

    """
    loads dataset into memory and performs centering and unit sphere projection of features

    Parameters
    ----------
    dataset_filename : name of file contianing extracted features of dataset (performs ensembling if list contains multiple filenames)
    train_c : number or training classes in dataset (used for centering and projection)
    device : cpu or cuda
    

    Returns
    -------
    dataset
    """

    dataset = torch.cat([torch.load(fn, map_location=torch.device(device)).to(device) for fn in dataset_filename], dim = 2)
    shape = dataset.shape
    average = dataset[:train_c].reshape(-1, dataset.shape[-1]).mean(dim = 0)

    dataset = dataset.reshape(-1, dataset.shape[-1]) - average
    dataset = dataset / torch.norm(dataset, dim = 1, keepdim = True)
    dataset = dataset.reshape(shape)
 
    return dataset



def generate_runs(n_runs, dataset, n_ways, n_samples, alpha_dir, gs, beta, kappa, m, train_c, test_c, val_c, device):
    
    """
    generates runs for active few-shot tasks with dirichlet class imbalance and and performs graph smoothing

    Parameters
    ----------
    n_runs : number of runs
    dataset : dataset used for tasks (after feature extraction)
    n_ways : number of classes
    n_samples : number of samples
    alpha_dir : dirichlet imbalance parameter
    gs : False to turn off graph smoothing
    beta, kappa, m : graph smoothing parameters 
    train_c : number of train classes in dataset
    val_c : number of validation classes in dataset
    test_c : number of test classes in dataset
    device : cpu or cuda
    

    Returns
    -------
    runs_x
        runs inputs
    runs_y
        runs targets
    """

    runs_x = torch.zeros(n_runs,n_samples,dataset.shape[-1]).to(device)
    runs_y = torch.zeros(n_runs,n_samples).to(device)
    dirs = get_dirichlet_query_dist(alpha_dir, n_runs,  n_ways, n_samples)
    
    for i in range(n_runs):
        classes = torch.randperm(test_c)[:n_ways] + train_c +val_c
        
        dq = 0 #dirichlet number of samples per class
        for c,j in enumerate(dirs[i]):
            runs_x[i,dq:dq+j]= dataset[classes[c]][torch.randperm(dataset.shape[1])[:j]]
            runs_y[i,dq:dq+j] += c 
            dq += j

    if gs==True:
        runs_x = graph_smoothing(runs_x, device, beta, kappa, m)
    
    return runs_x,runs_y

torch.manual_seed(0)


parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('-device', default='cuda',
                    help='path to dataset')
parser.add_argument('-dataset-name', default='stl10',
                    help='dataset name', choices=['stl10', 'cifar10'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('--ds', type=int, default=100,
                    help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('--n_labels', type=int, default=25,
                    help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')

feature_dim = 512

def wacc_acc_f1(runs_preds,runs_y,n_ways):

    """
    computes weighted accuracy, unweighted accuracy and weighted F1-score for predictions

    Parameters
    ----------
    runs_x : runs inputs
    runs_y : runs targets
    n_ways : number of ways
    

    Returns
    -------
    mat_avg
        average class means from label
    optional ~ shots_per_class 
        classes count in labels
    """

    acc = []
    f1 = []
    n_runs = runs_preds.shape[0]
    wacc = torch.zeros((n_runs,1))        
    for i in range(n_runs):
        acc.append(accuracy_score(runs_preds[i].cpu(),runs_y[i].cpu()))
        f1.append(f1_score(runs_preds[i].cpu(),runs_y[i].cpu(),average='weighted'))        
    
    for i in range(n_ways):
        class_pred = (runs_preds==i).float().cpu()
        class_true = (runs_y==i).float().cpu()
        wacc += torch.nan_to_num(((class_true*class_pred).sum(axis=1))/class_true.sum(axis=1),nan=1).reshape(-1,1)/n_ways
    wacc = wacc.reshape(-1).tolist()
    return wacc,acc,f1  


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, feature=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.fc is None:
            if feature:
                return x, None
            else:
                return x
        if feature:
            x1 = self.fc(x)
            return x, x1
        x = self.fc(x)
        return x


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def get_model(num_classes):
    return resnet18(num_classes=num_classes)

def load_resnet18_tiered():

    model = torch.nn.DataParallel(get_model(351)).cuda()
    checkpoint = torch.load('models/checkpoints/tiered/softmax/resnet18/model_best.pth.tar')
    model_dict = model.state_dict()
    params = checkpoint['state_dict']
    params = {k: v for k, v in params.items() if k in model_dict}
    model_dict.update(params)
    model.load_state_dict(model_dict)

    return model

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_checkpoint(state, is_best, filename='checkpointsmclr.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.writer = SummaryWriter()
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)


        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")
        print("Start SimCLR training for "+str(self.args.epochs)+" epochs.")

        for epoch_counter in range(self.args.epochs):
            for images, _ in tqdm(train_loader):
                images = torch.cat(images, dim=0)

                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    trf = without_augment()
                    for i in range(images.shape[0]):
                        images[i] = trf(pil(images[i]))
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))


                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 5:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")
            print('Epoch : '+str(epoch_counter)+' \nLoss : '+str(loss)+' \nTop acc : '+str(top1[0]))

        logging.info("Training has finished.")
        print('Training done')
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
        print('saved at ',self.writer.log_dir)
    

def without_augment(size=84, enlarge=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    if enlarge:
        resize = int(size*256./224.)
    else:
        resize = size
    return transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                normalize,
            ])

if __name__ == "__main__":
    accs = []
    for i in range(10):
        args = parser.parse_args()
        fs_classes = np.arange(160)
        np.random.shuffle(fs_classes)
        np.random.shuffle(fs_classes)
        fs_task_classes = fs_classes[:5].tolist()
        class_images =class_images_dict (fs_task_classes,dataset_path='.\\datasets\data\\tiered_imagenet\data\\')
        ds_size = args.ds
        
        dataset = ContrastiveLearningDataset()
        train_dataset = dataset.get_dataset(name='tiered', n_views=2, ds_size=ds_size, class_images=class_images)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=32, shuffle=True,
            num_workers=12, pin_memory=True, drop_last=True)

        runs_x = torch.zeros(ds_size,3,84,84)
        runs_xx = torch.zeros(ds_size,512)
        runs_y = torch.tensor(train_dataset.targets)
        model = load_resnet18_tiered()
        trf = without_augment()




        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                            last_epoch=-1)

        #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
        with torch.cuda.device(args.gpu_index):
            simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
            simclr.train(train_loader)

        #few_shot_task
        with torch.no_grad():
            model.eval()
            torch.cuda.empty_cache()
            for i in tqdm(range(ds_size)):
                im = read_image(train_dataset.im_paths[i])    
                runs_x[i]  = trf(pil(im)).float()
            for i in range(args.ds//64):

                runs_xx[i*64:(i+1)*64] = model(runs_x[i*64:(i+1)*64],True)[0]
            runs_xx[(i+1)*64:] = model(runs_x[(i+1)*64:],True)[0]
        
        labels_idces = torch.randperm(args.ds)[:args.n_labels]

        labels_x = runs_xx[labels_idces]
        labels_y = runs_y[labels_idces]

        mlp = MLP(64)
        mlp.fit(labels_x,labels_y.reshape(-1))
        y_pred = torch.tensor(mlp.predict(runs_xx)).unsqueeze(0)
        print('result is')
        acc = wacc_acc_f1(y_pred,runs_y.unsqueeze(0),5)[0]
        print(acc)
        accs.append(acc[0])
    print("best: {:.2f}% (± {:.2f}%)".format(stats(accs, "")[0]*100,stats(accs, "")[1]*100))