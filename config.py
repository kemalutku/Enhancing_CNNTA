from model import CnnTa
from datetime import datetime
import os

max_epochs = 1000
batch_size = 32
learning_rate = 1e-3

# Number of trading days used as the input window size. Increase this
# to train on longer historical contexts, e.g. 30, 60 or 90 days.
sequence_len = 15

working_dir = os.getcwd()

train_dirs = {
    "1d": os.path.join(working_dir, "data_finance", "train", "1d"),
    "1wk": os.path.join(working_dir, "data_finance", "train", "1wk"),
    "1mo": os.path.join(working_dir, "data_finance", "train", "1mo"),
}
test_dirs = {
    "1d": os.path.join(working_dir, "data_finance", "test", "1d"),
    "1wk": os.path.join(working_dir, "data_finance", "test", "1wk"),
    "1mo": os.path.join(working_dir, "data_finance", "test", "1mo"),
}

results_dir = os.path.join(working_dir, "results")
record_dir = os.path.join(working_dir, "records")
checkpoint_dir = os.path.join(working_dir, "checkpoints")

num_parallel_trainings = 1

model = CnnTa
in_channels = 3
class_weights = [1, 2, 2]

train_years = [2017, 2022]
test_years = [2023, 2024]

indicators = [
    "RSI", "WIL", "WMA", "EMA", "SMA", "HMA", "TMA", "CCI", "CMO", "MCD",
    "PPO", "ROC", "CMF", "ADX", "PSA",
]

run_name_base = r"DOW30_1h"
run_name = f"{run_name_base}-{datetime.now().strftime('%m_%d_%H_%M')}"
