import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from dataset import FashionDataset

# ✅ 路径设置
image_path = '/Users/mashang/Desktop/9444as2/【AiDLab】A100/LAT/image'

# ✅ 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 使用设备：{device}")

# ✅ 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ✅ 加载数据集
train_set = FashionDataset('train.csv', image_path, transform=transform)
val_set = FashionDataset('val.csv', image_path, transform=transform)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32)

# ✅ 获取类别数
num_classes = len(train_set.label2idx)
print(f"📊 类别数：{num_classes}")

# ✅ 加载预训练 VGG16 模型
vgg = models.vgg16(pretrained=True)

# 替换分类头
vgg.classifier[6] = nn.Linear(vgg.classifier[6].in_features, num_classes)
vgg = vgg.to(device)

# ✅ 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg.parameters(), lr=1e-4)

# ✅ 训练模型
best_acc = 0
for epoch in range(10):
    vgg.train()
    correct, total = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = vgg(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total

    # 验证
    vgg.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = vgg(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total

    print(f"📦 Epoch [{epoch+1}/10] | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    # 保存最优模型
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(vgg.state_dict(), 'best_model_vgg.pth')
        print("🎯 已保存最佳模型：best_model_vgg.pth")
