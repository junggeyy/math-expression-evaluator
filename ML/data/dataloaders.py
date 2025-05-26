import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import datasets, transforms

def get_dataset_loaders(data_dir="ML/data/dataset", batch_size=64):
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
    )

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    class_to_idx = dataset.class_to_idx

    print(f"Found {len(dataset)} images across {len(class_to_idx)} classes")
    print("Classes:", list(class_to_idx.keys()))

    torch.manual_seed(42)

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, 
                            [train_size, val_size, test_size])

    train_loader = DataLoader(
        dataset=train_dataset,
        num_workers=2,
        batch_size=batch_size,
        persistent_workers=True,
        shuffle=True
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader, class_to_idx