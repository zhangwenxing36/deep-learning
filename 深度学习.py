import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

# 1. 数据准备与增强
# 数据增强
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 设置训练集和验证集路径
train_dir = r"C:\Users\zhaozihao\PycharmProjects\pythonProject1\.venv\flowers"
test_dir = r"C:\Users\zhaozihao\PycharmProjects\pythonProject1\.venv\flowers"

# 使用ImageFolder来加载数据集
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 用于可视化图片的函数
def imshow(img):
    img = img / 2 + 0.5  # 去除归一化的影响
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # 转换维度为HWC格式
    plt.show()

# 展示原始数据和增强数据
def visualize_data(loader):
    data_iter = iter(loader)
    images, labels = next(data_iter)

    # 展示原始数据的第一张图片
    print("原始数据的第一张图片:")
    imshow(images[0])  # 展示第一个样本

    # 展示增强后的第一张图片
    # 由于transform是随机的，展示增强后的图片也是随机的
    print("增强后的第一张图片:")
    imshow(images[0])  # 展示经过增强的数据样本

# 展示数据增强的效果
visualize_data(train_loader)

# 2. 建立模型：使用预训练的ResNet18进行微调
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False  # 冻结卷积层参数

# 修改最后的全连接层，以适应花卉数据集的5个类别
model.fc = nn.Linear(model.fc.in_features, 5)

# 将模型移至CPU
device = torch.device("cpu")
model.to(device)

# 3. 定义损失函数与优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# 输出模型结构
from torchsummary import summary
summary(model, input_size=(3, 224, 224))  # 输入的图像尺寸是224x224

# 4. 训练模型
num_epochs = 20
train_losses = []
train_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 计算准确率
        _, predicted = torch.max(outputs, 1)
        total_preds += labels.size(0)
        correct_preds += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct_preds / total_preds

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

# 5. 绘制损失和准确率图
plt.figure(figsize=(12, 5))

# 损失图
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

# 准确率图
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.show()

# 6. 评估模型
model.eval()  # 切换到评估模式
all_labels = []
all_preds = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        # 预测
        _, predicted = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

# 评估指标公式说明
print("评估指标公式说明：")
print("Precision = TP / (TP + FP)  # 精度（Precision）公式")
print("Recall = TP / (TP + FN)     # 召回率（Recall）公式")
print("F1-score = 2 * (Precision * Recall) / (Precision + Recall)  # F1分数公式")

# 打印分类报告
print("Classification Report:")
print(classification_report(all_labels, all_preds))

# 混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
plt.title("Confusion Matrix")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# 误分类样本分析
print("误分类样本分析：")
misclassified_images = []

# 遍历测试集，找到误分类的样本
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        # 找到误分类样本
        misclassified = (predicted != labels)
        misclassified_images.extend(zip(inputs[misclassified], predicted[misclassified], labels[misclassified]))

# 可视化误分类样本
misclassified_images = misclassified_images[:5]  # 展示前5个误分类样本
for i, (img, pred, label) in enumerate(misclassified_images):
    img = img.cpu().numpy().transpose((1, 2, 0))  # 转为HWC格式
    plt.imshow(img)
    plt.title(f"Predicted: {train_dataset.classes[pred]}, True: {train_dataset.classes[label]}")
    plt.show()

# 7. 保存模型
torch.save(model.state_dict(), "flower_classifiers.pth")










