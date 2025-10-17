import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import time
from random import shuffle
from collections import Counter
import itertools
from sklearn.metrics import classification_report, confusion_matrix
import os
from PIL import Image

# 辛普森角色映射
map_characters = {
    0: 'homer_simpson',           # 2246张图片
    1: 'bart_simpson',            # 1342张图片
    2: 'lisa_simpson',           # 1354张图片
    3: 'marge_simpson',          # 1291张图片
    4: 'moe_szyslak',            # 1452张图片
    5: 'ned_flanders',           # 1454张图片
    6: 'principal_skinner',       # 1194张图片
    7: 'charles_montgomery_burns', # 1193张图片
    8: 'krusty_the_clown',       # 1206张图片
    9: 'milhouse_van_houten',     # 1079张图片
    10: 'sideshow_bob',          # 877张图片
    11: 'chief_wiggum',          # 986张图片
    12: 'apu_nahasapeemapetilon', # 623张图片
    13: 'kent_brockman',         # 498张图片
    14: 'comic_book_guy',        # 469张图片
    15: 'edna_krabappel',        # 457张图片
    16: 'nelson_muntz',          # 358张图片
    17: 'lenny_leonard'          # 310张图片
}

# 超参数设置
batch_size = 32
epochs = 50
num_classes = len(map_characters)
pictures_per_class = 300  # 每个角色最多使用300张图片
test_size = 0.15
learning_rate = 0.001

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class SimpsonDataset(Dataset):
    """辛普森数据集类"""
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)

def load_real_pictures():
    """
    从characters文件夹加载真实图片数据
    """
    pics = []
    labels = []
    
    print("加载真实辛普森数据集...")
    
    for k, char in map_characters.items():
        char_path = f'./characters/{char}'
        if not os.path.exists(char_path):
            print(f"警告: 角色 {char} 的文件夹不存在，跳过...")
            continue
            
        # 获取该角色的所有图片
        pictures = glob.glob(f'{char_path}/*.jpg')
        
        if len(pictures) == 0:
            print(f"警告: 角色 {char} 没有找到图片，跳过...")
            continue
        
        # 限制每个角色的图片数量
        nb_pic = min(pictures_per_class, len(pictures))
        selected_pictures = np.random.choice(pictures, nb_pic, replace=False)
        
        print(f"加载角色 {char}: {nb_pic} 张图片")
        
        for pic_path in selected_pictures:
            try:
                # 使用PIL读取图片
                img = Image.open(pic_path)
                
                # 转换为RGB（如果不是的话）
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                pics.append(img)
                labels.append(k)
                
            except Exception as e:
                print(f"读取图片失败 {pic_path}: {e}")
                continue
    
    print(f"总共加载了 {len(pics)} 张图片")
    return pics, np.array(labels)

def get_data_transforms():
    """获取数据变换"""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def get_dataset():
    """创建数据集"""
    # 加载图片
    images, labels = load_real_pictures()
    
    if len(images) == 0:
        print("错误: 没有加载到任何图片数据！")
        return None, None, None, None
    
    # 分割数据集
    from sklearn.model_selection import train_test_split
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=test_size, random_state=42, stratify=labels)
    
    # 获取数据变换
    train_transform, val_transform = get_data_transforms()
    
    # 创建数据集
    train_dataset = SimpsonDataset(train_images, train_labels.astype(np.int64), train_transform)
    test_dataset = SimpsonDataset(test_images, test_labels.astype(np.int64), val_transform)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"测试集: {len(test_dataset)} 样本")
    
    # 显示每个类别的样本数量
    train_counts = Counter(train_labels)
    test_counts = Counter(test_labels)
    
    print("\n训练集和测试集分布:")
    for k, char in map_characters.items():
        train_count = train_counts.get(k, 0)
        test_count = test_counts.get(k, 0)
        print(f"{char}: {train_count} 训练样本, {test_count} 测试样本")
    
    return train_loader, test_loader

class ResNetSimpson(nn.Module):
    """基于ResNet的辛普森角色识别模型"""
    def __init__(self, num_classes=18):
        super(ResNetSimpson, self).__init__()
        
        # 加载预训练的ResNet18
        self.resnet = models.resnet18(pretrained=True)
        
        # 冻结前面的层
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # 替换最后的全连接层
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.resnet(x)

def train_model(model, train_loader, test_loader, epochs=50):
    """训练模型"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.1)
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        # 计算平均损失和准确率
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        scheduler.step()
        
        print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, train_accs, val_losses, val_accs

def evaluate_model(model, test_loader):
    """评估模型"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # 计算准确率
    correct = sum(1 for p, t in zip(all_preds, all_targets) if p == t)
    accuracy = 100. * correct / len(all_targets)
    
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    # 分类报告
    print('\nClassification Report:')
    print(classification_report(all_targets, all_preds, 
                              target_names=list(map_characters.values())))
    
    # 混淆矩阵
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(15, 12))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('ResNet Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(map_characters))
    plt.xticks(tick_marks, list(map_characters.values()), rotation=45, ha='right')
    plt.yticks(tick_marks, list(map_characters.values()))
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # 在矩阵中添加数值
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black", fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    return accuracy

def plot_training_history(train_losses, train_accs, val_losses, val_accs):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 准确率图
    ax1.plot(train_accs, label='Train Accuracy')
    ax1.plot(val_accs, label='Validation Accuracy')
    ax1.set_title('ResNet Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(True)
    
    # 损失图
    ax2.plot(train_losses, label='Train Loss')
    ax2.plot(val_losses, label='Validation Loss')
    ax2.set_title('ResNet Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    """主函数"""
    print("开始辛普森角色识别训练...")
    print("使用ResNet和真实数据集...")
    
    # 获取数据集
    train_loader, test_loader = get_dataset()
    
    if train_loader is None:
        print("数据集加载失败，退出程序")
        return None
    
    # 创建模型
    model = ResNetSimpson(num_classes=num_classes).to(device)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 训练模型
    print("开始训练...")
    model, train_losses, train_accs, val_losses, val_accs = train_model(
        model, train_loader, test_loader, epochs=epochs)
    
    # 评估模型
    print("\n评估模型...")
    accuracy = evaluate_model(model, test_loader)
    
    # 绘制训练历史
    plot_training_history(train_losses, train_accs, val_losses, val_accs)
    
    # 保存模型
    torch.save(model.state_dict(), 'simpson_resnet_model.pth')
    print("模型已保存为 'simpson_resnet_model.pth'")
    
    return model

if __name__ == '__main__':
    model = main()
