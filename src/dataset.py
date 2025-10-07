import os
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset

class BrainMRIDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        label = int(self.img_labels.iloc[idx, 1])

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (128, 128))
        image = image.astype('float32') / 255.0
    
        if self.transform:
            image = self.transform(image)

        image = torch.tensor(image).permute(2, 0, 1)
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label
