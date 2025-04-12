import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset_AAT import FashionDataset_AAT  # 修改此处导入

# ----------------------------
# ✅ 设置路径和设备
# ----------------------------
image_path = '/Users/liujiaxiang/Desktop/9444gs/【AiDLab】A100/AAT/image'
csv_file = 'train_b.csv'
val_file = 'val_b.csv'
model_path = 'best_model_cnn_b.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 使用设备：{device}")

# ----------------------------
# ✅ CNN 模型定义（和 Task A 相同）
# ----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Feature block 1
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Feature block 2
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),  # Flatten to feed into linear layer

            nn.Linear(64 * 56 * 56, 512),  # FC layer 1
            nn.ReLU(),

            nn.Linear(512, num_classes)   # Output layer
        )

    def forward(self, x):
        return self.model(x)

# ----------------------------
# ✅ 图像预处理
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.3),  # 稍微保守一点
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ----------------------------
# ✅ 加载数据
# ----------------------------
train_set = FashionDataset_AAT(csv_file, image_path, transform=transform, label_type='sub')
val_set = FashionDataset_AAT(val_file, image_path, transform=transform, label_type='sub')
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64)  # 更大一些

num_classes = len(train_set.label2idx)
print(f"📊 子类别数：{num_classes}")

# ----------------------------
# ✅ 初始化模型、损失函数、优化器
# ----------------------------
model = SimpleCNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ----------------------------
# ✅ 训练模型
# ----------------------------
epochs = 20
best_acc = 0

for epoch in range(epochs):
    model.train()
    running_loss = 0
    correct_train, total_train = 0, 0  # 👈 新增变量统计训练准确率

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # 👇 统计训练集准确率
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    train_acc = correct_train / total_train  # 👈 计算训练准确率

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

    # 👇 修改输出格式：插入 Train Acc
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), model_path)
        print("✅ 最佳模型已保存")

print("✅ 训练完成！")
