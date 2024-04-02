import math
from types import new_class
from torch.optim.lr_scheduler import LambdaLR


def SinelineLR(optimizer, warmup_epochs, max_epochs):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch) / float(warmup_epochs)
        else:
            progress = (current_epoch - warmup_epochs) / (max_epochs - warmup_epochs)
            return 0.5 * (1.0 + math.sin(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    return scheduler


import numpy as np
import torch.nn as nn
from ignite.metrics import Accuracy, Loss, Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced


class NewData(Metric):
    def __init__(
        self, trainer, output_transform=lambda x: x, device="cuda", threshold=0.9
    ):
        self.new_data = []
        self.trainer = trainer
        self.p = threshold
        super(NewData, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self.new_data = []
        super(NewData, self).reset()

    @sync_all_reduce("new_data")
    def compute(self):
        data = np.asarray(self.new_data, dtype=object)
        cnt = len(self.new_data)
        print(f"returning {cnt} pesdo-labels from test set")
        return (data, cnt)

    @reinit__is_reduced
    def update(self, output):
        y_pred, x = output[2].detach(), output[0].detach()
        softmax = nn.Softmax(dim=1)
        y_pred = softmax(y_pred)
        y_pred = y_pred.cpu().numpy()
        x_test = x.cpu().numpy()
        for i in range(y_pred.shape[0]):
            p_max = y_pred[i].max()
            label = np.argmax(y_pred[i])
            if p_max >= self.p:
                data = [x_test[i], label]
                self.new_data.append(data)


import torch
from torch.utils.data import Subset


class CustomSubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.new_data = []
        self.new_data_len = 0

    def __getitem__(self, idx):
        if idx >= len(self.indices):
            return (
                torch.from_numpy(self.new_data[idx - len(self.indices)][0]),
                self.new_data[idx - len(self.indices)][1],
            )
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices) + self.new_data_len

    def add(self, new_data):
        self.new_data_len = new_data.shape[0]
        self.new_data = np.asarray(new_data)
        for i in range(self.new_data_len):
            self.new_data[i][0] = np.float32(self.new_data[i][0])


import os


def clear_folder(folder_path):
    if not os.path.exists(folder_path):
        return

    if not os.path.isdir(folder_path):
        return

    if os.listdir(folder_path):
        # 文件夹不为空，清空文件夹
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                clear_folder(file_path)
    else:
        # 文件夹为空
        print("文件夹为空")
