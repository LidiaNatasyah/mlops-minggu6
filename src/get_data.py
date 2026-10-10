import torchvision
import os

# Unduh dataset Fashion-MNIST ke folder 'data'
print("Mengunduh Fashion-MNIST...")
train_data = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True)
test_data = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True)
print("Selesai! Data tersimpan di folder ./data/FashionMNIST")