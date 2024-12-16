from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import fiftyone as foz

class FOImageDataset(Dataset):
    def __init__(self, filepaths, transform=None):
        self.filepaths = filepaths
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path
    

train_dataset_fo = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=None,
    max_samples=10000,
    classes=["Home appliance"],
    dataset_name="home_appliance_dataset",
    download_if_necessary=True,
)

val_dataset_fo = foz.load_zoo_dataset(
    "open-images-v7",
    split="validation",
    label_types=None,
    classes=["Home appliance"],
    max_samples=100,
    dataset_name="home_appliance_dataset",
    download_if_necessary=True
)

train_filepaths = [sample.filepath for sample in train_dataset_fo]
val_filepaths = [sample.filepath for sample in val_dataset_fo]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_dataset = FOImageDataset(train_filepaths, transform=train_transform)
val_dataset = FOImageDataset(val_filepaths, transform=val_transform)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)


