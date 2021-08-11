from .datasets.chess import Chess_Dataset
from .datasets.fire import Fire_Dataset

dataset_factory = {
    "chess": Chess_Dataset,
    "fire" : Fire_Dataset
}

def get_dataset(dataset):
    return dataset_factory[dataset]