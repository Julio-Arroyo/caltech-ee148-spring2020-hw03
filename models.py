import torch.nn as nn


class Net(nn.Module):
    '''
    Build the best MNIST classifier.
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.feature_extractor = nn.Sequential(
                            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(8),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=2, stride=2),

                            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(8),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=2, stride=2),
                            nn.Dropout(0.3),
                            
                            nn.Flatten(),
                            nn.Linear(8*7*7, 300), # for full size
                            #  nn.Linear(288, 300), # when center crop size=24
                            nn.ReLU(),
                            nn.Linear(300, 100),
                            nn.ReLU(),
                            # PyTorch implementation of cross-entropy loss includes softmax layer
                        )
        self.linear = nn.Linear(100, 10)

    def forward(self, x):
        feature_vec = self.feature_extractor(x)
        return self.linear(feature_vec)

    def extract_features(self, x):
        return self.feature_extractor(x)
