import os

from PIL import Image
from torchvision.datasets import VisionDataset


class Dataset(VisionDataset):
    def __init__(self, root, transform=None):
        super().__init__(transform)
        self.root = root
        self.transform = transform
        self.samples = []
        for c_id, c in enumerate(["Baked Potato", "Taco"]):
            c_path = os.path.join(self.root, c)
            for file in os.scandir(c_path):
                self.samples.append((file.path, c_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label