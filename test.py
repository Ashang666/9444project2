import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import FashionDataset
import os

# ✅ 图像路径
image_path = '/Users/mashang/Desktop/9444as2/【AiDLab】A100/LAT/image'

# ✅ 模型保存路径
model_path = 'best_model.pth'

# ✅ 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 使用设备：{device}")

# ✅ 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ✅ 加载测试集
test_set = FashionDataset('test.csv', image_path, transform=transform)
test_loader = DataLoader(test_set, batch_size=32)

# ✅ 获取类别数 & 类别映射（用于打印）
num_classes = len(test_set.label2idx)
idx2label = {v: k for k, v in test_set.label2idx.items()}

# ✅ 模型定义（要和训练用的一样）
import torch.nn as nn
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# ✅ 加载模型
model = SimpleCNN(num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ✅ 开始评估
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

test_acc = correct / total
print(f"\n✅ 测试集准确率：{test_acc:.4f}（共 {total} 张图像）")
