"""
==========================
Illustration of transforms
==========================

This example illustrates the various transforms available in :ref:`the
torchvision.transforms module <transforms>`.
"""

# sphinx_gallery_thumbnail_path = "../../gallery/assets/transforms_thumbnail.png"

from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as transforms


plt.rcParams["savefig.bbox"] = 'tight'
orig_img = Image.open(Path('assets') / 'astronaut.jpg')
# if you change the seed, make sure that the randomly-applied transforms
# properly show that the image can be both transformed and *not* transformed!
torch.manual_seed(0)


def plot(imgs, with_orig=True, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [orig_img] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.show()



# ###################################
# # AutoAugment
# # ~~~~~~~~~~~
# # The :class:`~torchvision.transforms.AutoAugment` transform
# # automatically augments data based on a given auto-augmentation policy.
# # See :class:`~torchvision.transforms.AutoAugmentPolicy` for the available policies.
# policies = [T.AutoAugmentPolicy.CIFAR10,
#             T.AutoAugmentPolicy.IMAGENET, T.AutoAugmentPolicy.SVHN]
# augmenters = [T.AutoAugment(policy) for policy in policies]
# imgs = [
#     [augmenter(orig_img) for _ in range(12)]
#     for augmenter in augmenters
# ]
# row_title = [str(policy).split('.')[-1] for policy in policies]
# plot(imgs, row_title=row_title)

# ####################################
# # RandAugment
# # ~~~~~~~~~~~
# # The :class:`~torchvision.transforms.RandAugment` transform automatically augments the data.
# augmenter = T.RandAugment()
# imgs = [augmenter(orig_img) for _ in range(4)]
# plot(imgs)

# ####################################
# # TrivialAugmentWide
# # ~~~~~~~~~~~~~~~~~~
# # The :class:`~torchvision.transforms.TrivialAugmentWide` transform automatically augments the data.
# augmenter = T.TrivialAugmentWide()
# imgs = [augmenter(orig_img) for _ in range(4)]
# plot(imgs)

# ####################################
# # AugMix
# # ~~~~~~
# # The :class:`~torchvision.transforms.AugMix` transform automatically augments the data.
# augmenter = T.AugMix()
# imgs = [augmenter(orig_img) for _ in range(4)]
# plot(imgs)


# ####################################
# # RandomApply
# # ~~~~~~~~~~~
# # The :class:`~torchvision.transforms.RandomApply` transform
# # randomly applies a list of transforms, with a given probability.
# applier = T.RandomApply(transforms=[T.RandomCrop(size=(64, 64))], p=0.5)
# transformed_imgs = [applier(orig_img) for _ in range(4)]
# plot(transformed_imgs)


affine_transfomer = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(p=0.36),
    transforms.RandomSolarize(threshold=192.0, p=0.2),
    transforms.RandomInvert(p=0.36),
    transforms.RandomEqualize(p = 0.5),
    transforms.ElasticTransform(alpha=50.0),
    transforms.RandomAffine(degrees=(-45, 45), translate=(0.1, 0.3), scale=(0.6, 1.4)),
])
affine_imgs = [affine_transfomer(orig_img) for _ in range(12)]
plot(affine_imgs)