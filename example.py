import torch
import torch.nn as nn
from ShapeTracer import ShapeTracer

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 10),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block(x)
        x = self.head(x)
        return x

model = TinyCNN()
x = torch.randn(4, 3, 224, 224)

tracer = ShapeTracer(model)
with tracer:
    y = model(x)