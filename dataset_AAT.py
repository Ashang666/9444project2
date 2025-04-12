import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class FashionDataset_AAT(Dataset):
    def __init__(self, csv_file, image_dir, transform=None, label_type='main'):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

        # 选择使用主类还是子类标签列
        if label_type == 'main':
            label_column = 'label_main'
        elif label_type == 'sub':
            label_column = 'label_sub'
        else:
            raise ValueError("label_type must be 'main' or 'sub'")

        # 将类别名转换为数字编码
        self.label2idx = {label: idx for idx, label in enumerate(sorted(self.data[label_column].unique()))}
        self.data['label'] = self.data[label_column].map(self.label2idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = self.data.iloc[idx]['label']

        if self.transform:
            image = self.transform(image)

        return image, label
