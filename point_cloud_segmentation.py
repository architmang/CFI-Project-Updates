import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import open3d as o3d

import torch
from torch.utils.data import DataLoader

from learning3d.models import PointNet, Classifier
from learning3d.losses import ClassificationLoss
from learning3d.data_utils import ClassificationData, ModelNet40Data

if __name__ == '__main__':
    ptnet = PointNet(emb_dims=1024, input_shape='bnc', use_bn=True)
    model = Classifier(feature_model=ptnet)

    trainset = ClassificationData(data_class=ModelNet40Data(train=True))
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)

    optimizer = torch.optim.Adam(model.parameters())

    max_epochs = 300
    for epoch in range(max_epochs):
        for batch_idx, data in enumerate(tqdm(trainloader)):
            points, target = data
            target = target.squeeze(-1)

            output = model(points)
            loss = ClassificationLoss()  # Pass arguments as keywords
            loss.forward(output, target)

            optimizer.zero_grad()
            optimizer.step()
