import torch
import lightning as L
import torch.nn.functional as F
import torchmetrics

class MLP(L.LightningModule):
    def __init__(self, model, learning_rate, num_classes):
        super().__init__()

        self.model = model
        self.learning_rate = learning_rate

        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x): 
        return self.model(x)
    
    def get_probas(self, batch):
        features, true_labels = batch
        logits = self(features)

        loss = F.cross_entropy(logits, true_labels)
        predicted_labels = torch.argmax(logits, dim=1)

        return loss, true_labels, predicted_labels
    
    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self.get_probas(batch)

        self.log("Train Loss", loss)
        self.train_accuracy(predicted_labels, true_labels)
        self.log(
            "train_acc", self.train_accuracy, prog_bar=True, on_epoch=True, on_step=False
        )

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self.get_probas(batch)

        self.log("Val Loss", loss)
        self.val_accuracy(predicted_labels, true_labels)
        self.log(
            "val_acc", self.val_accuracy, prog_bar=True
        )

        return loss
    
    def test_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self.get_probas(batch)

        self.log("Test Loss", loss)
        self.test_accuracy(predicted_labels, true_labels)
        self.log(
            "test_acc", self.test_accuracy, prog_bar=True
        )

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer