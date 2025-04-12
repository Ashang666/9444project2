import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models
import torch.nn as nn
from dataset_AAT import FashionDataset_AAT

# ✅ 设置路径
image_path = '/Users/liujiaxiang/Desktop/9444gs/【AiDLab】A100/AAT/image'
test_csv = 'test_b.csv'
model_path = 'best_model_vgg16_b.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 使用设备：{device}")

# ✅ 图像预处理（必须与训练保持一致）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ✅ 加载测试集
test_set = FashionDataset_AAT(test_csv, image_path, transform=transform, label_type='sub')
test_loader = DataLoader(test_set, batch_size=64)
num_classes = len(test_set.label2idx)
idx2label = {v: k for k, v in test_set.label2idx.items()}

# ✅ 加载模型（结构要和训练一致）
vgg16 = models.vgg16(weights=None)  # 不加载预训练权重
vgg16.classifier[6] = nn.Sequential(
    nn.Linear(4096, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, num_classes)
)
model = vgg16.to(device)

# ✅ 加载训练好的参数
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ✅ 测试评估
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
