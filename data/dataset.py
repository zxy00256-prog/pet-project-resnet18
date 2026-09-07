from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders():
    # 训练集数据增强
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 测试集预处理
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 适配你的目录：pet_project/data/oxford‑iiit‑pet/
    train_dataset = datasets.OxfordIIITPet(
        root="./data/petdata",
        split="trainval",
        download=False,
        transform=train_transform
    )

    test_dataset = datasets.OxfordIIITPet(
        root="./data/petdata",
        split="test",
        download=False,
        transform=test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders()
    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"测试集样本数: {len(test_loader.dataset)}")
    img, label = next(iter(train_loader))
    print(f"图片shape: {img.shape}, 标签shape: {label.shape}")