import numpy as np

class ReLU:
    def forward(self, x):
        return np.maximum(0, x)
    def __call__(self, x):
        return self.forward(x)

class Sigmoid:
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

class Tanh:
    def forward(self, x):
        return np.tanh(x)

