import os
import torch
from lightning import Trainer
from ML.models.mlp import MLP
from ML.models.pytorch_mlp import PyTorchMLP
from ML.data.dataloaders import get_dataset_loaders
import lightning as L

def train():
    L.seed_everything(42)

    # loading the data
    train_loader, val_loader, test_loader, class_to_idx = get_dataset_loaders()

    # defining our model
    num_features = 28 * 28
    num_classes = len(class_to_idx)
    
    print(f"Training model with {num_features} features and {num_classes} classes")

    base_model = PyTorchMLP(num_features, num_classes)
    lightning_model = MLP(model=base_model, learning_rate=0.0005, num_classes=num_classes)

    # defining Trainer
    trainer = Trainer(
        max_epochs=20,
        accelerator="auto",
        devices="auto",
    )

    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    train_acc = trainer.test(dataloaders=train_loader)[0]["test_acc"]
    val_acc = trainer.test(dataloaders=val_loader)[0]["test_acc"]
    test_acc = trainer.test(dataloaders=test_loader)[0]["test_acc"]
    print(
        f"Train Acc {train_acc*100:.2f}%"
        f" | Val Acc {val_acc*100:.2f}%"
        f" | Test Acc {test_acc*100:.2f}%"
    )

    os.makedirs("saved_models", exist_ok=True)
    torch.save(base_model.state_dict(), "saved_models/mlp-symbols.pt")
    print("Model weights saved to saved_models/mlp-symbols.pt")

if __name__ == "__main__":
    train()