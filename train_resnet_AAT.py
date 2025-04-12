# ✅ 修改版 ResNet18 训练代码，增强泛化能力，提高测试准确率

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from dataset_AAT import FashionDataset_AAT
import matplotlib.pyplot as plt

# ----------------------------
# ✅ 设置路径和设备
# ----------------------------
image_path = '/Users/liujiaxiang/Desktop/9444gs/【AiDLab】A100/AAT/image'
csv_file = 'train_b.csv'
val_file = 'val_b.csv'
model_path = 'best_model_resnet18_b.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 使用设备：{device}")

# ----------------------------
# ✅ 图像预处理（加入更丰富的数据增强）
# ----------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ----------------------------
# ✅ 加载数据
# ----------------------------
train_set = FashionDataset_AAT(csv_file, image_path, transform=train_transform, label_type='sub')
val_set = FashionDataset_AAT(val_file, image_path, transform=val_transform, label_type='sub')
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64)

num_classes = len(train_set.label2idx)
print(f"📊 子类别数：{num_classes}")

# ----------------------------
# ✅ 模型：ResNet18 + Dropout
# ----------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, num_classes)
)
model = model.to(device)

# ----------------------------
# ✅ 损失函数、优化器 + L2 正则项
# ----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

# ----------------------------
# ✅ 训练模型
# ----------------------------
epochs = 20
best_acc = 0

# ✅ 记录每一轮准确率
train_acc_list = []
val_acc_list = []

for epoch in range(epochs):
    model.train()
    running_loss = 0
    correct_train, total_train = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    train_acc = correct_train / total_train

    # 验证
    model.eval()
    correct_val, total_val = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)

    val_acc = correct_val / total_val

    # ✅ 记录准确率
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)

    print(f"📦 Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), model_path)
        print("✅ 最佳模型已保存")

print("🎉 训练完成！")
