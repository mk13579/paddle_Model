import paddle
import paddle.nn.functional as F
import numpy as np

class IMDBDataset(paddle.io.Dataset):
    def __init__(self, sents, labels):

        self.sents = sents
        self.labels = labels

    def __getitem__(self, index):

        data = self.sents[index]
        label = self.labels[index]

        return data, label

    def __len__(self):

        return len(self.sents)