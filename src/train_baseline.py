import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 1. Setup MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("fashion-mnist-experiment")

# 2. Definisikan Model CNN Sederhana
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 14 * 14, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(-1, 16 * 14 * 14)
        x = self.fc(x)
        return x

# 3. Load Data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# 4. Inisialisasi Model & Konfigurasi
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 3

# 5. Mulai Eksperimen di MLflow
with mlflow.start_run(run_name="cnn_baseline"):
    # Log parameter
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", 64)
    mlflow.log_param("model_type", "SimpleCNN")
    
    # Training Loop
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(trainloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss}")
        
        # Log metrik per epoch
        mlflow.log_metric("train_loss", avg_loss, step=epoch)

    # Ambil 1 contoh batch data untuk merekam format input
    example_input, _ = next(iter(trainloader))
    
    # Simpan model ke MLflow dengan menyertakan input_example
    mlflow.pytorch.log_model(model, name="model", input_example=example_input.numpy())
    print("Training baseline selesai dan tercatat di MLflow!")