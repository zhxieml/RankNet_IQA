import torch.nn as nn
import torchvision.models as models

class RankNet(nn.Module):
    def __init__(self, num_features):
        super(RankNet, self).__init__()

        # Use a simple Ranknet to learn the ranking.
        self.model = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Output a probability.
        self.output = nn.Sigmoid()

    def forward(self, input1, input2):
        s1 = self.model(input1)
        s2 = self.model(input2)
        prob = self.output(s1 - s2)

        return prob

class ResRankNet(nn.Module):
    def __init__(self, num_features):
        super(ResRankNet, self).__init__()

        # Use Resnet50 as the backbone to extract features.
        resnet50 = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet50.children())[:-1])
        self.middle = nn.Linear(resnet50.fc.in_features, num_features)

        # User RankNet to learn to rank.
        self.ranknet = RankNet(num_features)

    def forward(self, input1, input2):
        feat1 = self.backbone(input1)
        feat2 = self.backbone(input2)
        feat1, feat2 = feat1.view(feat1.size(0), -1), feat2.view(feat2.size(0), -1)
        feat1, feat2 = self.middle(feat1), self.middle(feat2)
        prob = self.ranknet(feat1, feat2)

        return prob