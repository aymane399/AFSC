from collections import defaultdict
import numpy as np
import random
import os
import torch
from torch import nn
from torchvision.transforms import transforms
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import Dataset

pil = transforms.ToPILImage()

np.random.seed(0)

def class_images_dict(fs_task_classes = [0,1,2,3,4],dataset_path='.\\datasets\data\\tiered_imagenet\data\\'):
    class_images = defaultdict(list)

    for file in os.listdir(dataset_path):
        
        filename = os.fsdecode(file) 
        if filename.startswith("test"): 
            if filename[8] == '_':
                f = filename[:8]
            elif filename[9] == '_':
                f = filename[:9]
            elif filename[10] == '_':
                f = filename[:10]
            elif filename[11] == '_':
                f = filename[:11]
            
            if int(f[7:]) in fs_task_classes:
                class_images[fs_task_classes.index(int(f[7:]))].append(dataset_path+filename)
            #tiered_testimages_path.append(filename)
            #tiered_test_classes.append(int(f[7:])) 
    return class_images

class tieredDataset(Dataset):
    def __init__(self,ds_size, class_images , transform=None, target_transform=None):
        self.num_classes = len(class_images)
        self.samples_per_class = ds_size//self.num_classes
        self.class_images = class_images
        self.transform = transform
        self.target_transform = target_transform
        self.lends = ds_size
        for i in range(self.num_classes):
            random.shuffle(self.class_images[i])
            self.class_images[i] = self.class_images[i][:self.samples_per_class]
        self.im_paths = []
        self.targets = []
        for key in self.class_images:
            for im_p in self.class_images[key]:
                self.im_paths.append(im_p)
                self.targets.append(key)
    def __len__(self):
        return self.lends

    def __getitem__(self, idx):
        
        label = self.targets[idx]
        img_path = self.im_paths[idx]
        image = pil(read_image(img_path))
        if self.transform:
            image = self.transform(image)

        return image, label

class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img

class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]

class ContrastiveLearningDataset:
    def __init__(self):
        self.root_folder = '.\\datasets\data\\tiered_imagenet\data\\'

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, name, n_views, ds_size, class_images):
        valid_datasets = {'tiered': lambda: tieredDataset(  ds_size, 
                                                            class_images, 
                                                            transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(84),
                                                                  n_views), 
                                                            target_transform=None)}

        dataset_fn = valid_datasets[name]
        return dataset_fn()
