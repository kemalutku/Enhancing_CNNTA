from model import *
from datetime import datetime
import os

max_epochs = 200
batch_size = 128
learning_rate = 1e-3

train_dir = r"C:\Users\KemalUtkuLekesiz\Documents\Kod\Tez\CnnTezV3\pythonProject\data\train\1d\all_train_2002_2016.csv"
test_dir = r"C:\Users\KemalUtkuLekesiz\Documents\Kod\Tez\CnnTezV3\pythonProject\data\test\1d"
results_dir = r"C:\Users\KemalUtkuLekesiz\Documents\Kod\Tez\CnnTezV3\pythonProject\results"

train_index_dir = r"C:\Users\KemalUtkuLekesiz\Documents\Kod\Tez\CnnTezV3\pythonProject\data\train\index\all_train_2002_2025.csv"
test_index_dir = r"C:\Users\KemalUtkuLekesiz\Documents\Kod\Tez\CnnTezV3\pythonProject\data\test\index\DJI_2017_2017_test.csv"

checkpoint_dir = r"C:\Users\KemalUtkuLekesiz\Documents\Kod\Tez\CnnTezV3\pythonProject\checkpoints"

model = CnnTa
apply_bn = True
use_index = False

# indicators = ['RSI', 'WIL', 'WMA', 'EMA', 'SMA', 'HMA', 'TMA', 'CCI', 'CMO', 'MCD', 'PPO', 'ROC', 'CMF', 'ADX',
#               'TN_RSI', 'PSA', 'STOCHRSI', 'UO', 'MOM', 'TSI', 'AO', 'KDJ', 'RVI', 'VIDYA', 'ZLEMA', 'DEMA', 'T3',
#               'ATR', 'BBW', 'DONCH']

indicators = ["RSI", "WIL", "WMA", "EMA", "SMA", "HMA", "TMA", "CCI", "CMO", "MCD", "PPO", "ROC", "CMF", "ADX",
              "PSA"]

# indicators = ["RSI", "ROC", "MCD", "HMA", "SMA", "CCI", "PPO", "ATR", "BBW", "OBV", "MFI", "STOCHRSI", "ADX", "TRIX",
#               "PSA"]

window_length = len(indicators)

run_name = f"SET2IND-{datetime.now().strftime('%Y_%m_%d_%H_%M')}"

hparams = {
    "lr": learning_rate,
    "batch_size": batch_size,
    "model": "CNN-TA",
    "optimizer": "Adam",
    "loss": "Focal Loss",
    "activation": "gelu",
    "class_weights": "[1, 2, 2]",
    "Batch Normalization": apply_bn,
    "Normalization": False,
    "Training Dataset": "DOW30",
    "Training Years": "2002 - 2016",
    "Testing Years": "2017 - 2017",
    "Labeling Method": "FTH",
    "Labeling Method Parameters": "Horizon: 5, threshold: 0.02"
}
