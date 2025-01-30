import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

# 데이터 로드 및 전처리
data_path = "/Users/kimjinha/Documents/GitHub/youngflow/pitchain 2/data/ratings_small.csv"

data = pd.read_csv(data_path,encoding='utf-8')

# 입력 열 (viewtime.1, viewtime.2, ...)
X_columns = [col for col in data.columns if col.startswith('viewed time')]
X = data[X_columns].values

# 출력 열 (like click.1 + ctr.1, ...)
like_click_columns = [col for col in data.columns if col.startswith('like click')]
ctr_columns = [col for col in data.columns if col.startswith('ctr')]
y = data[like_click_columns].values + data[ctr_columns].values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 텐서로 변환
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# DataLoader 생성
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 모델 정의
class ViewTimePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ViewTimePredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.model(x)

# 모델 초기화
input_dim = X_train.shape[1]  # 입력 feature 개수
hidden_dim = 128
output_dim = y_train.shape[1]  # 출력 feature 개수

model = ViewTimePredictor(input_dim, hidden_dim, output_dim)

# 손실 함수 및 옵티마이저 정의
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, criterion, optimizer, epochs=100,save_path="model_weights.pth"):
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            writer.add_scalar("Loss/train", loss, epoch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        #print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

        #가중치 저장
        torch.save(model.state_dict(), save_path)
        #print(f"Model weights saved to {save_path}")

# 모델 학습
train_model(model, train_loader, criterion, optimizer, epochs=100,save_path="model_weights.pth")
writer.flush()
writer.close()

def load_model(model, load_path="model_weights.pth"):
    model.load_state_dict(torch.load(load_path))
    print(f"Model weights loaded from {load_path}")
    return model

# 모델 평가
def evaluate_model(model, test_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            writer.add_scalar("Loss/", loss)
            total_loss += loss.item()
    print(f"Test Loss: {total_loss / len(test_loader):.4f}")

load_model(model,load_path="model_weights.pth")
evaluate_model(model, test_loader)

model.eval()

with torch.no_grad():  # 그래디언트 계산 비활성화
    print("x input tensor : ", X_train_tensor)
    output = model(X_train_tensor[0])
    print("Model Output:", output)
    print("y output tensor : ", y_train[0])