import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random

# 1. Setup MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("fashion-mnist-experiment")

# Parameter Terbaik dari Optuna
BEST_LR = 0.002022849302522856
BEST_BATCH = 32
BEST_DROPOUT = 0.01290588966952233
SEEDS = [42, 43, 44] # 3 Seed berbeda sesuai instruksi tugas

# Fungsi untuk mengunci seed agar reproducible
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class SimpleCNN(nn.Module):
    def __init__(self, dropout_rate):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 14 * 14, 10)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(-1, 16 * 14 * 14)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def train_and_evaluate(seed):
    set_seed(seed)
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BEST_BATCH, shuffle=True)
    
    model = SimpleCNN(BEST_DROPOUT)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=BEST_LR)
    
    epochs = 3
    final_loss = 0.0
    
    with mlflow.start_run(run_name=f"best_model_seed_{seed}", nested=True):
        mlflow.log_params({"lr": BEST_LR, "batch_size": BEST_BATCH, "dropout_rate": BEST_DROPOUT, "seed": seed})
        
        for epoch in range(epochs):
            running_loss = 0.0
            for data in trainloader:
                inputs, labels = data
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            final_loss = running_loss / len(trainloader)
            mlflow.log_metric("train_loss", final_loss, step=epoch)
            
        # Simpan model untuk seed ini
        example_input, _ = next(iter(trainloader))
        mlflow.pytorch.log_model(model, name="model", input_example=example_input.numpy())
        
    return final_loss

if __name__ == "__main__":
    print("Memulai pelatihan dengan 3 seed berbeda...")
    loss_results = []
    
    with mlflow.start_run(run_name="evaluasi_3_seed"):
        for seed in SEEDS:
            print(f"Melatih model dengan Seed: {seed}")
            loss = train_and_evaluate(seed)
            loss_results.append(loss)
            
        mean_loss = np.mean(loss_results)
        std_loss = np.std(loss_results)
        
        mlflow.log_metric("mean_loss", mean_loss)
        mlflow.log_metric("std_loss", std_loss)
        
        print("\n=== HASIL EVALUASI ===")
        print(f"Loss 3 Seed (Mean ± Std): {mean_loss:.4f} ± {std_loss:.4f}")
        print("Tugas uji stabilitas 3 seed selesai!")