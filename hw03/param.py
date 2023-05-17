import torchvision.transforms as transforms
import torch
import torch.nn as nn


_exp_name = "sample"

# set a random seed for reproducibility
myseed = 6666  

# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"

# The number of batch size.
batch_size = 64

# The number of training epochs
pre_epochs = 100
n_epochs = 10

# If no improvement in 'patience' epochs, early stop.
patience = 100




# Normally, We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# However, it is also possible to use augmentation in the testing phase.
# You may use train_tfm to produce a variety of images and then test using ensemble methods
pre_train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize((128, 128)),
    # You may add some transforms here.
    transforms.RandomHorizontalFlip(p=0.36),
    transforms.RandomSolarize(threshold=192.0, p=0.2),
    transforms.RandomInvert(p=0.36),
    transforms.RandomEqualize(p = 0.5),
    transforms.ElasticTransform(alpha=50.0),
    transforms.RandomAffine(degrees=(-45, 45), translate=(0.1, 0.3), scale=(0.6, 1.4)),
    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor(),
])

train_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
    transforms.ToTensor(),
])