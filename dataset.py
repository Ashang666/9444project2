import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FashionDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

        # 将类别名转换为数字编码
        self.label2idx = {label: idx for idx, label in enumerate(sorted(self.data['label'].unique()))}
        self.data['label'] = self.data['label'].map(self.label2idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = self.data.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label

'''
你之后可以这样调用它（写在 train.py 或 main.py）：
from dataset import FashionDataset
from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = FashionDataset(csv_file='train.csv', image_dir='path_to_image_folder', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
'''
