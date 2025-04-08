import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models
from dataset import FashionDataset

# ✅ 设置路径
image_path = '/Users/mashang/Desktop/9444as2/【AiDLab】A100/LAT/image'
csv_file = 'test.csv'
model_path = 'best_model_resnet.pth'

# ✅ 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 使用设备：{device}")

# ✅ 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ✅ 加载测试集
test_set = FashionDataset(csv_file, image_path, transform=transform)
test_loader = DataLoader(test_set, batch_size=32)
num_classes = len(test_set.label2idx)
idx2label = {v: k for k, v in test_set.label2idx.items()}

# ✅ 加载 ResNet 模型
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# ✅ 模型测试
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"✅ 测试集准确率：{accuracy:.4f}（共 {total} 张图像）")
