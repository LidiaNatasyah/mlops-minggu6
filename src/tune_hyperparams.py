import optuna
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 1. Setup MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("fashion-mnist-experiment")

# 2. Definisikan Model CNN dengan Dropout dinamis
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

# 3. Fungsi Objective untuk Optuna
def objective(trial):
    # Tentukan rentang hyperparameter yang mau dicari
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)

    # Gunakan nested=True agar rapi di bawah parent run
    with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
        mlflow.log_params({"lr": lr, "batch_size": batch_size, "dropout_rate": dropout_rate})
        
        # Load Data
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

        model = SimpleCNN(dropout_rate)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Latih singkat (2 epoch) per trial agar tidak terlalu lama
        epochs = 2
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
            
            avg_loss = running_loss / len(trainloader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

        return avg_loss # Optuna akan mencari nilai loss terkecil

if __name__ == "__main__":
    # Parent Run untuk merangkum semua trial Optuna
    with mlflow.start_run(run_name="optuna-tuning-20-trials"):
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=20) # Syarat minimal 20 trial
        
        # Log hasil parameter terbaik ke Parent Run
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_loss", study.best_value)
        print(f"\nTuning selesai! Parameter terbaik: {study.best_params}")