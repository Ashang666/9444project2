import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from dataset_AAT import FashionDataset_AAT

# ----------------------------
# ✅ 设置路径和设备
# ----------------------------
image_path = '/Users/liujiaxiang/Desktop/9444gs/【AiDLab】A100/AAT/image'
csv_file = 'train_b.csv'
val_file = 'val_b.csv'
model_path = 'best_model_vgg16_b.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 使用设备：{device}")

# ----------------------------
# ✅ 图像预处理
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])  # VGG16 官方推荐值
])

# ----------------------------
# ✅ 加载数据
# ----------------------------
train_set = FashionDataset_AAT(csv_file, image_path, transform=transform, label_type='sub')
val_set = FashionDataset_AAT(val_file, image_path, transform=transform, label_type='sub')
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64)

num_classes = len(train_set.label2idx)
print(f"📊 子类别数：{num_classes}")

# ----------------------------
# ✅ 加载预训练 VGG16 模型并替换最后一层
# ----------------------------
vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)  # 加载预训练权重
# 解冻部分后面的卷积层
for name, param in vgg16.features.named_parameters():
    if '28' in name or '26' in name:  # 最后两层卷积
        param.requires_grad = True
    else:
        param.requires_grad = False


vgg16.classifier[6] = nn.Sequential(
    nn.Linear(4096, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, num_classes)
)

model = vgg16.to(device)

# ----------------------------
# ✅ 损失函数与优化器
# ----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# ----------------------------
# ✅ 训练模型
# ----------------------------
epochs = 20
best_acc = 0
for epoch in range(epochs):
    model.train()
    running_loss = 0
    correct_train, total_train = 0, 0  # 👈 新增用于计算训练准确率

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

    train_acc = correct_train / total_train  # 👈 计算 Train Acc

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

    # ✅ 输出：加上 Train Acc
    print(f"📦 Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), model_path)
        print("✅ 最佳模型已保存")

print("🎉 训练完成！")
