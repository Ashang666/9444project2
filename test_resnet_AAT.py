import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models
from dataset_AAT import FashionDataset_AAT
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn

# ✅ 设置路径
image_path = '/Users/liujiaxiang/Desktop/9444gs/【AiDLab】A100/AAT/image'
test_csv = 'test_b.csv'
model_path = 'best_model_resnet18_b.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 使用设备：{device}")

# ✅ 图像预处理（和训练时一致）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ✅ 加载测试集（使用 AAT 版本的 Dataset）
test_set = FashionDataset_AAT(test_csv, image_path, transform=transform, label_type='sub')
test_loader = DataLoader(test_set, batch_size=32)
num_classes = len(test_set.label2idx)
idx2label = {v: k for k, v in test_set.label2idx.items()}

# ✅ 模型定义（结构需和训练一致）
model = resnet18(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(512, num_classes)
)

model = model.to(device)

# ✅ 加载训练好的模型
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
