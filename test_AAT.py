import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset_AAT import FashionDataset_AAT
import torch.nn as nn
import os

# ✅ 设置路径和设备
image_path = '/Users/liujiaxiang/Desktop/9444gs/【AiDLab】A100/AAT/image'
test_csv = 'test_b.csv'
model_path = 'best_model_cnn_b.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 使用设备：{device}")

# ✅ 图像预处理（要和训练时一致）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ✅ 加载测试集
test_set = FashionDataset_AAT(test_csv, image_path, transform=transform, label_type='sub')
test_loader = DataLoader(test_set, batch_size=64)
num_classes = len(test_set.label2idx)
idx2label = {v: k for k, v in test_set.label2idx.items()}

# ✅ 模型结构（与训练保持一致）
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# ✅ 加载模型权重
model = SimpleCNN(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ✅ 评估测试集准确率
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

acc = correct / total
print(f"\n✅ 测试集准确率：{acc:.4f}（共 {total} 张图像）")
