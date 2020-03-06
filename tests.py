import torch

from nets import BaseEncoder

if __name__ == '__main__':

    base_encoder = BaseEncoder()
    print(base_encoder.backbone)
    print(base_encoder.projection_head)

    rand_input = torch.zeros((1, 3, 224, 224)).random_()
    embeddings = base_encoder(rand_input)