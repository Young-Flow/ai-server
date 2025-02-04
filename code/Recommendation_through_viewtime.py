import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import os
from typing import List
from model_data import MemberPreferenceInfoRes, PreferenceInfoRes

def convert_to_tenser_data(infos: List[MemberPreferenceInfoRes]):
    """
    Convert user preference data into a structured tensor dataset.
    - Sort data by `memberId` (column-wise) and `bmId` (row-wise).
    - Drop `memberId` and `bmId` before training.(학습 데이터가 아니므로 드랍)
    """
    data = []

    for info in infos:
        member_id = info.memberId
        for preference_info in info.preferenceInfoResList:
            bm_id = preference_info.bmId
            view_time = preference_info.spViewTime  # Feature
            ctr = int(preference_info.isViewed)  # Convert `isViewed` (True/False) to 1/0
            like_click = int(preference_info.isInvested)  # Convert `isInvested` (True/False) to 1/0
            target = ctr + like_click  # Label calculation

            # Store (memberId, bmId, feature, target) for sorting
            data.append([member_id, bm_id, view_time, target])

    # Convert to Pandas DataFrame for easier sorting
    df = pd.DataFrame(data, columns=["memberId", "bmId", "viewTime", "target"])

    df = df.sort_values(by=["memberId", "bmId"]).reset_index(drop=True)
    df["viewTime"] = (df["viewTime"] - df["viewTime"].mean()) / df["viewTime"].std()
    # Drop `memberId` and `bmId` (used only for sorting)
    df_final = df.drop(columns=["memberId", "bmId"])

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(df_final[["viewTime"]].values, dtype=torch.float32)
    y_tensor = torch.tensor(df_final[["target"]].values, dtype=torch.float32)

    # Train-Test Split
    train_size = int(0.8 * len(X_tensor))
    test_size = len(X_tensor) - train_size

    indices = torch.randperm(len(X_tensor))  # Shuffle indices
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    X_train_tensor = X_tensor[train_indices]
    X_test_tensor = X_tensor[test_indices]
    y_train_tensor = y_tensor[train_indices]
    y_test_tensor = y_tensor[test_indices]

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, df  # Keep `df` for reference


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

def train_model(writer, model, train_loader, criterion, optimizer, epochs=100,save_path="model_weights.pth"):
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

        #가중치 저장
        torch.save(model.state_dict(), save_path)

def load_model(infos: List[MemberPreferenceInfoRes]):
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, df_mapping = convert_to_tenser_data(infos)

    # DataLoader 생성
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    input_dim = X_train_tensor.shape[1]  # 입력 feature 개수
    hidden_dim = 128
    output_dim = y_train_tensor.shape[1]  # 출력 feature 개수

    model = ViewTimePredictor(input_dim, hidden_dim, output_dim)

    # 손실 함수 및 옵티마이저 정의
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 모델 학습
    writer = SummaryWriter()
    train_model(writer, model, train_loader, criterion, optimizer, epochs=100)

    # 모델 평가
    load_path="model_weights.pth"
    model.load_state_dict(torch.load(load_path))
    print(f"Model weights loaded from {load_path}")
    evaluate_model(writer, model, test_loader, criterion, X_train_tensor, y_train_tensor)
    model.eval()

    writer.flush()
    writer.close()
    return model, X_train_tensor, y_train_tensor, df_mapping

# 모델 평가
def evaluate_model(writer, model, test_loader, criterion, X_train_tensor, y_train_tensor):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            writer.add_scalar("Loss/test", loss)
            total_loss += loss.item()
    print(f"Test Loss: {total_loss / len(test_loader):.4f}")