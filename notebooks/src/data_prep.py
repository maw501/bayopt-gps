import gzip
import pickle
from pathlib import Path

import requests
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

MNIST_H, MNIST_W = 28, 28
dev = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

# torchvision datasets are PILImage images of range [0, 1] => transform to
# Tensors of normalized range [-1, 1]
mnist_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)


def download_mnist(folder="data"):
    DATA_PATH = Path(folder)
    PATH = DATA_PATH / "mnist"
    PATH.mkdir(parents=True, exist_ok=True)
    URL = "http://deeplearning.net/data/mnist/"
    FILENAME = "mnist.pkl.gz"

    if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

    with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(
            f, encoding="latin-1"
        )

    x_train, y_train, x_valid, y_valid = map(
        torch.tensor, (x_train, y_train, x_valid, y_valid)
    )

    train_ds = TensorDataset(x_train, y_train)
    valid_ds = TensorDataset(x_valid, y_valid)
    return train_ds, valid_ds


def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )


def preprocess(x, y):
    return x.view(-1, 1, MNIST_H, MNIST_W).to(dev), y.to(dev)


def to_gpu(x, y):
    return x.to(dev), y.to(dev)


class WrapDL:
    """Generator applying pre-processing function to a batch as it's yielded"""

    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))
