# from torchvision.datasets import VOCDetection
# from pprint import pprint
# import cv2
# import numpy as np
from typing import final

import torch
from networkx.algorithms.isomorphism.tests.test_vf2pp import labels_many
from networkx.algorithms.threshold import weights_to_creation_sequence
from torch.utils.data.datapipes.utils.decoder import imagespecs
from torch.xpu import device
from torchvision.transforms import ToTensor
from VOCDataset import VOCDataset
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
# weight duoc train tren bo coco
# doi mo hinh resnet sang mobile net vi qua to
from torch.utils.data import DataLoader

def collate_fn(batch):
    images, labels = zip(*batch)
    # anh chi can la list cua tensor
    return images, labels

def train():
    print(torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))

    transform = ToTensor()
    train_dataset = VOCDataset(root="my_pascal_VOC", year="2012", image_set="train", download=False, transform=transform)
    # image, target = train_dataset[2000]
    # print(image.shape)
    # print(target)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )
    model = fasterrcnn_mobilenet_v3_large_fpn(weight=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT).to(device)
    model.train()

    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)

    for images, labels in train_dataloader:
        # print(images)
        # print(targets)
        # break

        #tuple khong co to to Tensor nen can chuyen nhu the nay
        images = [image.to(device) for image in images]
        labels = [{"boxes": target["boxes"].to(device), "labels": target["labels"].to(device)} for target in labels]

        losses = model(images, labels)
        #in ra loss cua classification va regression - RPN

        # final_losses = sum(loss for loss in losses.values())
        final_losses = torch.stack(list(losses.values())).sum()
        print(final_losses.item() )

        #Back ward
        optimizer.zero_grad() # lam sach buffer
        final_losses.backward()
        optimizer.step()

    #moi co dataset chi lay ra tung anh can lay ra dataloader
    #dataLoader can stack nhung anh co cung kich thuoc

if __name__ == '__main__':
    train()
