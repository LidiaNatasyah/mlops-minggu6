import mlflow
from mlflow.tracking import MlflowClient
import torch
import torchvision
import torchvision.transforms as transforms

# 1. Setup MLflow & Client
mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()
MODEL_NAME = "fashion-mnist-cnn"

# 2. Cari run_id dari evaluasi seed terbaik sebelumnya
# Kita ambil run uji stabilitas terakhir yang sukses
runs = mlflow.search_runs(experiment_names=["fashion-mnist-experiment"], filter_string="tags.mlflow.runName LIKE 'best_model_seed_%'", order_by=["metrics.train_loss ASC"], max_results=1)
best_run_id = runs.iloc[0]["run_id"]
model_uri = f"runs:/{best_run_id}/model"

print(f"Menggunakan run_id terbaik: {best_run_id}")

# 3. Daftarkan model ke MLflow Model Registry
mv = mlflow.register_model(model_uri, MODEL_NAME)
print(f"Model berhasil didaftarkan dengan Versi: {mv.version}")

# 4. Beri alias @champion
client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
print(f"Model versi {mv.version} kini diset sebagai '@champion'!")

# 5. Uji muat model dari alias @champion untuk inferensi
print("Menguji muat model via alias @champion...")
loaded_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")

# Ambil sampel data untuk uji prediksi
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform)
sample_input, sample_label = testset[0]

# Ubah dimensi tensor agar sesuai format batch
input_tensor = sample_input.unsqueeze(0).numpy()
preds = loaded_model.predict(input_tensor)

print(f"Prediksi kelas model untuk data uji: {preds}")
print(f"Label asli data uji: {sample_label}")
print("Alur End-to-End MLOps Selesai Sempurna!")