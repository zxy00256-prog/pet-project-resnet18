import os
import torch
import torch.nn as nn
import torch.optim as optim
from models.model import get_resnet18
from data.dataset import get_dataloaders


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    # 从dataset.py获取loader，不再在此处写数据集代码
    train_loader, val_loader = get_dataloaders()

    num_classes = 37
    model = get_resnet18(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    epochs = 10

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, pred = torch.max(outputs.data, dim=1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
        val_acc = correct / total

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss:{train_loss/len(train_loader):.4f} | Val Acc:{val_acc:.4f}")

    os.makedirs("./checkpoint", exist_ok=True)
    torch.save(model.state_dict(), "./checkpoint/best.pth")
    print("训练完成，权重保存至 ./checkpoint/best.pth")


if __name__ == "__main__":
    main()