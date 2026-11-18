import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import interp1d
import os
import time

# 1. Environment and physical parameter configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_nodes = np.array([0, 24.0, 44.0, 63.6, 84.4, 170.0])


# 2. Model structure definition
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return torch.nn.functional.silu(x + self.net(x))


class PINN_ResNet(nn.Module):
    def __init__(self, in_dim=13, out_dim=12, h_dim=256):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(in_dim, h_dim), nn.SiLU())
        self.res_layers = nn.Sequential(*[ResBlock(h_dim) for _ in range(6)])
        self.output_layer = nn.Linear(h_dim, out_dim)

    def forward(self, x):
        return self.output_layer(self.res_layers(self.input_layer(x)))


# 3. Data load
input_cols = ['T', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6']
target_cols = ['H1', 'Q1', 'H2', 'Q2', 'H3', 'Q3', 'H4', 'Q4', 'H5', 'Q5', 'H6', 'Q6']
branch_cols = ['Hb4', 'Qb4', 'Hb5', 'Qb5', 'Hb19', 'Qb19', 'Hb11', 'Qb11', 'Hb20', 'Qb20']

def load_data_for_scaler():
    train_list = []
    for i in range(1, 9):
        fname = f'DATA - Test{i}.csv'
        if os.path.exists(fname):
            df = pd.read_csv(fname)
            train_list.append(df)
        else:
            print(f"Warning : No training data file {fname} found")

    if not train_list:
        raise FileNotFoundError("No found (DATA - Test1.csv ~ Test8.csv)！")

    train_all = pd.concat(train_list, ignore_index=True)

    sc_x, sc_y, sc_b = MinMaxScaler(), MinMaxScaler(), MinMaxScaler()
    sc_x.fit(train_all[input_cols].values)
    sc_y.fit(train_all[target_cols].values)
    sc_b.fit(train_all[branch_cols].values)

    return sc_x, sc_y

sc_x, sc_y = load_data_for_scaler()


# 4. Load the trained model
model_path = 'trained_MS-PINN.pth'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"The trained model file is not found: {model_path}")

model = PINN_ResNet(in_dim=len(input_cols), out_dim=12).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


# 5. Perform predictions for Test9 ~ Test11
output_dir = 'Predictions'
os.makedirs(output_dir, exist_ok=True)
start_t = time.time()

for test_idx in range(9, 12):
    fname = f'DATA - Test{test_idx}.csv'
    if not os.path.exists(fname):
        continue

    test_raw = pd.read_csv(fname)
    X_test_s = sc_x.transform(test_raw[input_cols].values)

    with torch.no_grad():
        X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to(device)
        Y_test_pred = sc_y.inverse_transform(model(X_test_t).cpu().numpy())

    T_raw = test_raw['T'].values
    H_pred_nodes = Y_test_pred[:, 0::2]
    Q_pred_nodes = Y_test_pred[:, 1::2]

    t_target = np.arange(0, 3601, 10)
    x_target = np.arange(0, 171, 10)

    f_H_t = interp1d(T_raw, H_pred_nodes, axis=0, bounds_error=False, fill_value="extrapolate")
    f_Q_t = interp1d(T_raw, Q_pred_nodes, axis=0, bounds_error=False, fill_value="extrapolate")
    H_t_target = f_H_t(t_target)
    Q_t_target = f_Q_t(t_target)

    f_H_x = interp1d(x_nodes, H_t_target, axis=1, bounds_error=False, fill_value="extrapolate")
    f_Q_x = interp1d(x_nodes, Q_t_target, axis=1, bounds_error=False, fill_value="extrapolate")
    H_grid = f_H_x(x_target)
    Q_grid = f_Q_x(x_target)

    T_mesh, X_mesh = np.meshgrid(t_target, x_target, indexing='ij')
    grid_df = pd.DataFrame({
        'T': T_mesh.flatten(),
        'X': X_mesh.flatten(),
        'H': H_grid.flatten(),
        'Q': Q_grid.flatten()
    })

    grid_out_path = os.path.join(output_dir, f'Pred_Test{test_idx}.csv')
    grid_df.to_csv(grid_out_path, index=False)

print(f"Total time: {time.time() - start_t:.2f} s")