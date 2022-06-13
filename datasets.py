import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import PIL.Image as Image
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from args import args


"""
    Most of the code is reused from : https://github.com/mboudiaf/TIM for model loading

"""

feature_dim = 512


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

def warp_tqdm(data_loader, disable_tqdm):
    if disable_tqdm:
        tqdm_loader = data_loader
    else:
        tqdm_loader = tqdm(data_loader, total=len(data_loader))
    return tqdm_loader

class DatasetFolder(object):

    def __init__(self, root, split_dir, split_type, transform, out_name=False):
        assert split_type in ['train', 'test', 'val', 'query', 'support']
        split_file = os.path.join(split_dir, split_type + '.csv')
        assert os.path.isfile(split_file), split_file
        with open(split_file, 'r') as f:
            split = [x.strip().split(',') for x in f.readlines()[1:] if x.strip() != '']

        data, ori_labels = [x[0] for x in split], [x[1] for x in split]
        label_key = sorted(np.unique(np.array(ori_labels)))
        label_map = dict(zip(label_key, range(len(label_key))))
        mapped_labels = [label_map[x] for x in ori_labels]

        self.root = root
        self.transform = transform
        self.data = data
        self.labels = mapped_labels
        self.out_name = out_name
        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert os.path.isfile(os.path.join(self.root, self.data[index])), os.path.join(self.root, self.data[index])
        img = Image.open(os.path.join(self.root, self.data[index])).convert('RGB')
        label = self.labels[index]
        label = int(label)
        if self.transform:
            img = self.transform(img)
        if self.out_name:
            return img, label, self.data[index]
        else:
            return img, label, index

class CategoriesSampler(Sampler):

    def __init__(self, label, n_iter, n_way, n_shot, n_query):

        self.n_iter = n_iter
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query

        label = np.array(label)
        self.m_ind = []
        unique = np.unique(label)
        unique = np.sort(unique)
        for i in unique:
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_iter

    def __iter__(self):
        for i in range(self.n_iter):
            batch_gallery = []
            batch_query = []
            classes = torch.randperm(len(self.m_ind))[:self.n_way]
            for c in classes:
                l = self.m_ind[c.item()]
                pos = torch.randperm(l.size()[0])
                batch_gallery.append(l[pos[:self.n_shot]])
                batch_query.append(l[pos[self.n_shot:self.n_shot + self.n_query]])
            batch = torch.cat(batch_gallery + batch_query)
            yield batch

def get_optimizer(module, optimizer_name, nesterov, lr, weight_decay):
    OPTIMIZER = {'SGD': torch.optim.SGD(module.parameters(), lr=lr, momentum=0.9,
                                        weight_decay=weight_decay, nesterov=nesterov),
                 'Adam': torch.optim.Adam(module.parameters(), lr=lr, weight_decay=weight_decay)}
    return OPTIMIZER[optimizer_name]

def get_model(num_classes):
    return resnet18(num_classes=num_classes)


def get_dataloader(split, enlarge=True, num_workers=0, batch_size=32, disable_random_resize=False,
                   path = './datasets/data/mini_imagenet/data', split_dir=None, jitter=False, aug=False, shuffle=True, out_name=False,
                   sample=None):
    # sample: iter, way, shot, query
    transform = without_augment(84, enlarge=enlarge)
    sets = DatasetFolder(path, '.\datasets\data\split/'+args.dataset, split, transform, out_name=out_name)
    if sample is not None:
        sampler = CategoriesSampler(sets.labels, *sample)
        loader = DataLoader(sets, batch_sampler=sampler,
                            num_workers=num_workers, pin_memory=False)
    else:
        loader = DataLoader(sets, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers, pin_memory=False)
    return loader



def without_augment(size=84, enlarge=False):
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

    if args.dataset == 'mini':
        samples_per_class = 600
        data_path = './datasets/data/mini_imagenet'
    elif args.dataset == 'cub':
        samples_per_class = 60
        data_path = './datasets/data/cub/CUB_200_2011/images'
    elif args.dataset == 'tiered':
        samples_per_class = 1300
        data_path = './datasets/data/tiered_imagenet/data'

    model = torch.nn.DataParallel(get_model(args.train_c)).cuda()
    checkpoint = torch.load('models/checkpoints/'+args.dataset+'/softmax/resnet18/model_best.pth.tar')
    model_dict = model.state_dict()
    params = checkpoint['state_dict']
    params = {k: v for k, v in params.items() if k in model_dict}
    model_dict.update(params)
    model.load_state_dict(model_dict)

    train_loader = get_dataloader('train', path = data_path)
    test_loader = get_dataloader('test', path = data_path)

    model.eval()

    with torch.no_grad():
        all_features_train = []
        all_labels_train = []
        for i, (inputs, labels, _) in enumerate(warp_tqdm(train_loader, False)):
            inputs = inputs.to('cuda')
            outputs, _ = model(inputs, True)
            all_features_train.append(outputs.cpu())
            all_labels_train.append(labels)
        all_features_train = torch.cat(all_features_train, 0)
        all_labels_train = torch.cat(all_labels_train, 0)
        

        all_features_test = []
        all_labels_test = []
        for i, (inputs, labels, _) in enumerate(warp_tqdm(test_loader, False)):
            inputs = inputs.to('cuda')
            outputs, _ = model(inputs, True)
            all_features_test.append(outputs.cpu())
            all_labels_test.append(labels)
        all_features_test = torch.cat(all_features_test, 0)
        all_labels_test = torch.cat(all_labels_test, 0)

    FE_dataset = torch.ones(args.train_c+args.test_c,samples_per_class,feature_dim)
    class_dic = defaultdict(int)

    for i in range(all_labels_train.shape[0]):
        lab = all_labels_train[i].item()
        FE_dataset[lab,class_dic[lab]] = all_features_train[i]
        class_dic[lab]+=1
    for i in range(all_labels_test.shape[0]):
        lab = all_labels_test[i].item()+args.train_c
        FE_dataset[lab,class_dic[lab]] = all_features_test[i]
        class_dic[lab]+=1

    for key in class_dic:
        if class_dic[key]!=samples_per_class:
            fillers = samples_per_class - class_dic[key]
            FE_dataset[key,-fillers:] = FE_dataset[key,:fillers]

    torch.save(FE_dataset, './datasets/'+args.dataset+'18.pt')