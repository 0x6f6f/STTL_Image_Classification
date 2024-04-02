import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.transforms.autoaugment import AutoAugment, AutoAugmentPolicy
from torchvision.datasets import CIFAR10

# Define the transformations
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

auto_augment = transforms.Compose([
    AutoAugment(AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

basic_transform = transforms.ToTensor()

# Load the CIFAR10 dataset
trainset = CIFAR10(root='./data', train=True, download=True, transform=None)

# Get a single image from the dataset
image, label = trainset[0]

# Apply the transformation to the image
augmented_image = auto_augment(image)

# Convert the augmented image tensor to a numpy array
augmented_image = augmented_image.numpy()

# De-normalize the augmented image
unnormed_augmented_image = augmented_image * std + mean

# Visualize the original image and its augmented form
fig, axes = plt.subplots(1, 2)
axes[0].imshow(image)
axes[0].set_title('Original Image')
axes[1].imshow(augmented_image.transpose(1, 2, 0))
axes[1].set_title('Augmented Image')
plt.savefig('vis_aug_cifar10.png')