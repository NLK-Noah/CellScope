# src/data_loader.py

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# -------------------------
# 1️⃣ Définir les transformations
# -------------------------

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------
# 2️⃣ Créer la classe Dataset
# -------------------------

class CellDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # créer mapping label -> index
        self.classes = sorted(self.df['Category'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])

        # ouvrir image
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        label = self.class_to_idx[row['Category']]
        return image, label

# -------------------------
# 3️⃣ Créer les DataLoaders
# -------------------------

def get_dataloaders(data_dir="../data", batch_size=32):
    train_dataset = CellDataset(os.path.join(data_dir, "train.csv"),
                                os.path.join(data_dir, "train"),
                                transform=train_transforms)
    val_dataset = CellDataset(os.path.join(data_dir, "validation.csv"),
                              os.path.join(data_dir, "validation"),
                              transform=val_test_transforms)
    test_dataset = CellDataset(os.path.join(data_dir, "test.csv"),
                               os.path.join(data_dir, "test"),
                               transform=val_test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# -------------------------
# 4️⃣ Test rapide
# -------------------------

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    images, labels = next(iter(train_loader))
    print(f"Batch d'images: {images.shape}")
    print(f"Batch de labels: {labels}")
