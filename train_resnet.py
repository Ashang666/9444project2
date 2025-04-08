import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from dataset import FashionDataset

# ✅ 图像路径
image_path = '/Users/mashang/Desktop/9444as2/【AiDLab】A100/LAT/image'

# ✅ 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 使用设备：{device}")

# ✅ 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ✅ 加载数据
train_set = FashionDataset('train.csv', image_path, transform=transform)
val_set = FashionDataset('val.csv', image_path, transform=transform)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32)

# ✅ 类别数
num_classes = len(train_set.label2idx)
print(f"📊 类别数：{num_classes}")

# ✅ 加载预训练模型 ResNet18
model = models.resnet18(pretrained=True)

# 替换最后全连接层
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# ✅ 损失函数与优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ✅ 训练过程
best_acc = 0
for epoch in range(10):
    model.train()
    correct, total = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total

    # ✅ 验证阶段
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total

    print(f"📦 Epoch [{epoch+1}/10] | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_model_resnet.pth')
        print("🎯 已保存最佳模型：best_model_resnet.pth")
