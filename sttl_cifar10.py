from ctypes import sizeof
from hmac import new
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

# Define the ResNet model using the `timm` library
import timm
from timm.models.pvt_v2 import pvt_v2_b1

# custom utils
import copy
import numpy as np
from sttl_utils import NewData, SinelineLR, CustomSubset
from ignite.utils import setup_logger
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit

from ignite.contrib.handlers.tensorboard_logger import *
import matplotlib.pyplot as plt
import seaborn as sns

# Create a logger
tb_logger = TensorboardLogger(log_dir="/data/STTLBLS_rev/experimental/logs")

# model = timm.create_model('pvt_v2_b1',pretrained=True,pretrained_cfg_overlay=dict(file='/data/STTLBLS_rev/experimental/pvt_v2_b1.pth'), num_classes=10)
model = timm.create_model("resnetv2_50", pretrained=False, num_classes=10)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set hyperparameters
warmup_epochs = 5
num_epochs = 100
batch_size = 256
learning_rate = 1e-4

# Load CIFAR-10 dataset and apply transformations
transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Load the full CIFAR-10 dataset
dataset_path = "./data"
full_dataset = torchvision.datasets.CIFAR10(
    root=dataset_path, train=True, download=True, transform=transform
)

labels = [full_dataset[i][1] for i in range(len(full_dataset))]
ss = StratifiedShuffleSplit(n_splits=1, test_size=0.1)
train_indices, valid_indices = list(ss.split(np.array(labels)[:, np.newaxis], labels))[
    0
]

train_dataset = CustomSubset(full_dataset, train_indices)
val_dataset = CustomSubset(full_dataset, valid_indices)
test_dataset = torchvision.datasets.CIFAR10(
    root=dataset_path, train=False, download=True, transform=transform
)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
)

# Initialize the loss function and optimizer
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Move model to the device
model = model.to(device)
criterion = criterion.to(device)

# Create the trainer and evaluator using PyTorch Ignite
trainer = create_supervised_trainer(model, optimizer, criterion, device=device)

default_metrics = {"accuracy": Accuracy(), "loss": Loss(criterion)}
sttl_metrics = {
    "accuracy": Accuracy(output_transform=lambda x: (x[2], x[1])),
    "loss": Loss(criterion, output_transform=lambda x: (x[2], x[1])),
    "new_data": NewData(trainer),
}

train_evaluator = create_supervised_evaluator(
    model, metrics=default_metrics, device=device
)
val_evaluator = create_supervised_evaluator(
    model, metrics=default_metrics, device=device
)
test_evaluator = create_supervised_evaluator(
    model=model,
    metrics=sttl_metrics,
    device=device,
    output_transform=lambda x, y, y_pred: (x, y, y_pred),
)
test_evaluator.logger = setup_logger("Test Evaluator")

# Attach loggers to trainer
tb_logger.attach_output_handler(
    trainer,
    event_name=Events.ITERATION_COMPLETED,
    tag="training",
    output_transform=lambda loss: {"loss": loss},
)
# tb_logger.attach(
#     trainer,
#     event_name=Events.EPOCH_COMPLETED,
#     log_handler=GradsHistHandler(model)
# )
tb_logger.attach_output_handler(
    train_evaluator,
    event_name=Events.EPOCH_COMPLETED,
    tag="training",
    metric_names=["loss", "accuracy"],
    global_step_transform=global_step_from_engine(trainer),
)
# Attach loggers to evaluator
tb_logger.attach_output_handler(
    val_evaluator,
    event_name=Events.EPOCH_COMPLETED,
    tag="val",
    metric_names=["loss", "accuracy"],
    global_step_transform=global_step_from_engine(trainer),
)

lr_scheduler = SinelineLR(optimizer, warmup_epochs, num_epochs)

log_pesdolabels_cnt = []


# Define event handlers for trainer and evaluator
@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(engine):
    train_evaluator.run(train_loader)
    train_metrics = train_evaluator.state.metrics
    print(
        f"Training Results - Epoch: {engine.state.epoch} | "
        f"Accuracy: {train_metrics['accuracy']*100:.2f}% | "
        f"Loss: {train_metrics['loss']:.4f}"
    )

    val_evaluator.run(test_loader)
    metrics = val_evaluator.state.metrics
    print(
        f"Validation Results - Epoch: {engine.state.epoch} | "
        f"Accuracy: {metrics['accuracy']*100:.2f}% | "
        f"Loss: {metrics['loss']:.4f}"
    )

    lr_scheduler.step()


@trainer.on(Events.EPOCH_COMPLETED)
def update_dataset(engine):
    # run test dataset
    test_evaluator.run(test_loader)
    metrics = test_evaluator.state.metrics
    print(
        f"Test Results for STTL - Epoch: {engine.state.epoch} | "
        f"Accuracy: {metrics['accuracy']*100:.2f}% | "
        f"Loss: {metrics['loss']:.4f}"
    )

    # STTL
    new_data, cnt = metrics["new_data"]
    log_pesdolabels_cnt.append(cnt)
    temp_trainset = copy.deepcopy(train_dataset)
    temp_trainset.add(new_data)
    trainloader = DataLoader(
        temp_trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    trainer.set_data(trainloader)


# Run the training loop
trainer.run(train_loader, max_epochs=num_epochs)

fig_title = "Count of Pesdo-labels"
sns.lineplot(log_pesdolabels_cnt)
plt.xlabel("Iter")
plt.ylabel(fig_title)
plt.savefig(f"./vis/{fig_title}.png")
