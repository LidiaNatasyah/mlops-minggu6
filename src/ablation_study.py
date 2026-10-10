import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 1. Setup MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("fashion-mnist-experiment")

BEST_LR = 0.002
BEST_BATCH = 32

# 2. Model dengan opsi mematikan Dropout
class SimpleCNN(nn.Module):
    def __init__(self, use_dropout=True):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 14 * 14, 10)
        # Jika use_dropout False, layer ini jadi Identity (tidak melakukan apa-apa)
        self.dropout = nn.Dropout(0.25) if use_dropout else nn.Identity()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(-1, 16 * 14 * 14)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# 3. Fungsi untuk mengeksekusi Ablation
def run_ablation(run_name, use_aug, use_dropout):
    # Ablasi Augmentasi (RandomHorizontalFlip)
    if use_aug:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BEST_BATCH, shuffle=True)
    
    model = SimpleCNN(use_dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=BEST_LR)
    
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({"use_augmentation": use_aug, "use_dropout": use_dropout})
        
        for epoch in range(2): # 2 epoch cukup untuk melihat perbedaan loss
            running_loss = 0.0
            for data in trainloader:
                inputs, labels = data
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            avg_loss = running_loss / len(trainloader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
        
        print(f"[{run_name}] Loss akhir: {avg_loss:.4f}")

if __name__ == "__main__":
    print("Mulai Ablation Study...")
    run_ablation("ablation_full_model", use_aug=True, use_dropout=True)
    run_ablation("ablation_no_augmentation", use_aug=False, use_dropout=True)
    run_ablation("ablation_no_dropout", use_aug=True, use_dropout=False)
    print("Ablation Study Selesai!")