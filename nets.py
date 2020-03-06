import torch.nn as nn
import torchvision.models as models

class BaseEncoder(nn.Module):
    """Implementation of base encoder(uses Resnet50) as mentioned in the paper
    https://arxiv.org/pdf/2002.05709.pdf"""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-1])
        self.projection_head = nn.Sequential(
                        nn.Linear(2048, 512),
                        nn.ReLU(inplace=True),
                        nn.Linear(512, 128),
                        nn.ReLU(inplace=True)
                        )

    def forward(self, x):
        output = self.backbone(x)
        output = output.reshape(output.size(0), -1)
        embeddings = self.projection_head(output)

        return embeddings