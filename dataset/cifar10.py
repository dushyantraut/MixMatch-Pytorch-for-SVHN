import numpy as np
from PIL import Image

import torchvision
import torch


def get_length(svhn_base_dataset):
    count = 0
    for img,label in svhn_base_dataset:
        #print(type(label))
        #exit()
        count+=1
    return count


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        #print(inp.shape)
        out1 = self.transform(inp)
        #print(out1.shape)
        out2 = self.transform(inp)
        #print(out2.shape)
        #print('done double transform')
        return out1, out2


def get_train_test(base_dataset, max_len, type):
    if(type == 'train'):
        start = 0
        end = max_len - 10000
    else:
        start = max_len - 10000
        end = max_len


    train_dict = {'img' : [], 'target' : [] }
    count = start 
    for img, label in base_dataset:
        if(count < end ):
            train_dict['img'].append(np.array(img))
            train_dict['target'].append(label)
            count+=1
        else:
            break
    train_dict['img'] = np.array(train_dict['img'])
    train_dict['target'] = np.array(train_dict['target'])
    return train_dict



def get_cifar10(root, n_labeled,
                 transform_train=None, transform_val=None,
                 download=True):

    
    
    
    base_dataset = torchvision.datasets.SVHN(root, download=download)
    max_len = get_length(base_dataset)
    train_dataset = get_train_test(base_dataset, max_len, type = 'train')
    test_dataset = get_train_test(base_dataset, max_len, type = 'test')

    print(f"train dataset {train_dataset['img'].shape} test_dataset {test_dataset['img'].shape}")
    print(f"train dataset {train_dataset['target'].shape} test_dataset {test_dataset['target'].shape}")
    
    train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(train_dataset['target'], int(n_labeled/10))
    #send only which are present in train set
    #print(f'maxlen {max_len}')
    #print(f'train_labeled_idxs {len(train_labeled_idxs)} train_unlabeled_idxs {len(train_unlabeled_idxs)} val_idxs {len(val_idxs)}')
    #print(f'train labeled idxs {min(train_labeled_idxs)} max {max(train_labeled_idxs)}')
    #print(f'train unlabeled idxs {min(train_unlabeled_idxs)} max {max(train_unlabeled_idxs)}')
    #print(f'val_idxs labeled idxs {min(val_idxs)} max {max(val_idxs)}')
    #print(len(train_labeled_idxs), len(train_unlabeled_idxs))
    
    #exit()
    print('cifar label train label')
    train_labeled_dataset = CIFAR10_labeled(train_dataset, train_labeled_idxs, transform=transform_train)
    #print(train_labeled_dataset.__len__())
    print('cifar unlabel train unlabel')
    train_unlabeled_dataset = CIFAR10_unlabeled(train_dataset, train_unlabeled_idxs, transform=TransformTwice(transform_train))
    print(train_unlabeled_dataset.__len__())
    print(f'train unlabeldata {train_unlabeled_dataset.len}')
    print('cifar label val label')
    val_dataset = CIFAR10_labeled(train_dataset, val_idxs, transform=transform_val, download=True)
    #print(val_dataset.__len__())

    test_dataset = CIFAR10_labeled(test_dataset, transform=transform_val, download=True)

    #print (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)} #Val: {len(val_idxs)}")
    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset
    #exit()

def train_val_split(labels, n_labeled_per_class):
    labels = np.array(labels)
    #print('in train val split')
    #print(labels.shape)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    for i in range(10):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:-500])
        val_idxs.extend(idxs[-500:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs

cifar10_mean = (0.4914, 0.4822, 0.4465) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616) # equals np.std(train_set.train_data, axis=(0,1,2))/255


def normalize(x, mean=cifar10_mean, std=cifar10_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean*255
    x *= 1.0/(255*std)
    return x

def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target]) 

def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border)], mode='reflect')

class RandomPadandCrop(object):
    """Crop randomly the image.
    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, x):
        #print(f'before x shape {x.shape}')
        x = pad(x, 4)
        #print(f'after x shape {x.shape}')
        h, w = x.shape[1:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        x = x[:, top: top + new_h, left: left + new_w]

        return x

class RandomFlip(object):
    """Flip randomly the image.
    """
    def __call__(self, x):
        if np.random.rand() < 0.5:
            x = x[:, :, ::-1]

        return x.copy()

class GaussianNoise(object):
    """Add gaussian noise to the image.
    """
    def __call__(self, x):
        c, h, w = x.shape
        x += np.random.randn(c, h, w) * 0.15
        return x

class ToTensor(object):
    """Transform the image to tensor.
    """
    def __call__(self, x):
        x = torch.from_numpy(x)
        return x

class CIFAR10_labeled():

    def __init__(self, base_dataset, indexs=None,
                 transform=None, target_transform=None,
                 download=False):
        '''
        super(CIFAR10_labeled, self).__init__(root,
                 transform=transform, target_transform=target_transform,
                 download=download)
        '''

        if indexs is not None:
            self.img = base_dataset['img'][indexs]
            self.targets = base_dataset['target'][indexs]
            #print('hi am here')
        else:
            self.img = base_dataset['img']
        #print(self.img.shape)
        #print(self.targets.shape)
        self.img = transpose(normalize(self.img))
        self.transform = transform
        self.target_transform = target_transform
        #print(self.img.shape)
        len = self.img.shape[0]
        self.len = len
        #print(len)
        #self.len = self.targets.shape[0]

        
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.img[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    

class CIFAR10_unlabeled(CIFAR10_labeled):

    def __init__(self, base_dataset, indexs,
                 transform=None, target_transform=None,
                 download=False):
        
        super(CIFAR10_unlabeled, self).__init__(base_dataset, indexs,
                 transform=transform, target_transform=target_transform,
                 download=download)
        
        #print(self.targets.shape)
        #print(type(self.targets))
        self.targets = [-1 for i in range(self.targets.shape[0])]
        

    def __len__(self):
        return self.len
        