import torch.nn as nn
import torchvision.models as models

class SomeNet(nn.Module):
    def __init__(self, num_features):
        super(SomeNet, self).__init__()

        # self.model = nn.Sequential(
        #     nn.Linear(num_features, 512),
        #     nn.Dropout(0.2),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.Dropout(0.2),
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        #     nn.Dropout(0.2),
        #     nn.ReLU(),
        #     nn.Linear(128, 1)
        # )
        self.model = nn.Linear(num_features, 1)

        # Output a probability.
        self.output = nn.Sigmoid()

    def forward(self, input):
        out = self.model(input)
        prob = self.output(out)

        return prob

class ResSomeNet(nn.Module):
    def __init__(self):
        super(ResSomeNet, self).__init__()

        # Use Resnet50 as the backbone to extract features.
        resnet50 = models.resnet50(pretrained=True)
        conv1_weight = resnet50.conv1.weight.data.clone()
        resnet50.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        resnet50.conv1.weight.data[:, :3] = nn.Parameter(conv1_weight)
        resnet50.conv1.weight.data[:, 3:6] = nn.Parameter(conv1_weight)

        self.backbone = nn.Sequential(*list(resnet50.children())[:-1])

        # # Freeze the backbone.
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        # User RankNet to learn to rank.
        self.somenet = SomeNet(resnet50.fc.in_features)

    def forward(self, input):
        feat = self.backbone(input)
        feat = feat.view(feat.size(0), -1)
        prob = self.somenet(feat)

        return prob