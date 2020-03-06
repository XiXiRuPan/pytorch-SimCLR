import argparse

import torch
import torch.nn as nn

from nets import BaseEncoder
from dataset import get_data_loaders
from train import Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--train-path', help='path to train dataset')
parser.add_argument('--test-path', help='path to test dataset')
parser.add_argument('--model-path', help='path to save model weights')
parser.add_argument('--image-width', type=int, default=224, help='width of input image')
parser.add_argument('--image-height', type=int, default=224, help='height of input image')
parser.add_argument('--batch-size', type=int, default=256, help='batch size during training')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--use-gpu', action='store_true', help='option to use gpu')

args = parser.parse_args()



if __name__ == '__main__':


    image_shape = (args.image_height, args.image_width)
    train_loader, test_loader = get_data_loaders(
                            image_shape=image_shape,
                            train_path=args.train_path,
                            test_path=args.test_path,
                            train_batch_size=args.batch_size,
                            test_batch_size=args.batch_size
                            )
    criterion = nn.CrossEntropyLoss()
    base_encoder = BaseEncoder()
    optimizer = torch.optim.Adam(params=base_encoder.parameters(), lr=args.lr)
    trainer = Trainer(
                dataloaders=(train_loader, test_loader),
                net=base_encoder,
                criterion=criterion,
                optimizer=optimizer,
                num_epochs=args.epochs,
                use_gpu=args.use_gpu
            )
    trainer.fit()