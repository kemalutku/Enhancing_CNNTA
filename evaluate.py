import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
from dataset import FinanceImageDataset
from model.focal_loss import FocalLoss
from test import trading
from dataset.RawFinanceDataset import RawFinanceDataset
import config
import argparse
import joblib

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def evaluate_model_on_folder(model, folder_path, indicators, batch_size, device, scaler_path):
    model.eval()
    class_weights = torch.tensor([1, 2, 2]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    if scaler_path is not None:
        scaler = joblib.load(scaler_path)
    else:
        scaler = None

    trading_result = {
        'Final Value': 0,
        'Buy & Hold': 0,
        'Annualized Return (%)': 0,
        'Sharpe Ratio (daily)': 0,
        'Trades Performed': 0,
        'Trades Won': 0,
        'Percent of Success': 0,
        'Avg Profit per Trade (%)': 0,
        'Avg Trade Length (days)': 0,
        'Max Profit/Trade (%)': 0,
        'Max Loss/Trade (%)': 0,
        'Idle Ratio (%)': 0,
        'Precision': 0,
        'Recall': 0,
        'F1': 0,
    }

    csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    total_test_loss = 0.0

    for csv_file in csv_files:
        test_dataset = RawFinanceDataset(csv_file, indicators, use_fitted_scaler=True, fitted_scaler=scaler,
                                         start_year=2017, end_year=2017)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        all_predictions = []
        all_labels = []
        all_timestamps = []
        all_close_candles = []
        test_loss = 0.0
        batch_count = 0

        with torch.no_grad():
            for ts, closes, images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                batch_count += 1

                all_predictions.extend(torch.argmax(outputs, dim=-1).cpu().numpy().flatten())
                all_labels.extend(torch.argmax(labels, dim=-1).cpu().numpy().flatten())
                all_timestamps.extend(ts.flatten().cpu().numpy())
                all_close_candles.extend(closes.flatten().cpu().numpy())

        file_loss = test_loss / batch_count if batch_count > 0 else 0
        total_test_loss += file_loss / len(csv_files)

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='macro', zero_division=0
        )
        result, _ = trading(all_timestamps, all_close_candles, all_predictions)

        for key in result:
            trading_result[key] += result[key] / len(csv_files)

        trading_result['Precision'] += precision / len(csv_files)
        trading_result['Recall'] += recall / len(csv_files)
        trading_result['F1'] += f1 / len(csv_files)

    return total_test_loss, trading_result


def main(stock_folder, checkpoint_path, scaler_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = config.model(apply_bn=config.apply_bn, window_length=config.window_length).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, trading_result = evaluate_model_on_folder(
        model=model,
        folder_path=stock_folder,
        indicators=config.indicators,
        batch_size=config.batch_size,
        device=device,
        scaler_path=scaler_path
    )

    print(f"\n✅ Test Loss: {test_loss:.7f}")
    print("📈 Trading Metrics:")
    for key, value in trading_result.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    folder = r"C:\Users\KemalUtkuLekesiz\Documents\Kod\Tez\CnnTezV3\pythonProject\data\evaluate\dow30\1d"
    checkpoint_path = r"C:\Users\KemalUtkuLekesiz\Documents\Kod\Tez\CnnTezV3\pythonProject\checkpoints\W122-2025_04_03_11_43_checkpoint_epoch_199.pth"
    # scaler_path = r"C:\Users\KemalUtkuLekesiz\Documents\Kod\Tez\CnnTezV3\pythonProject\data\train\1d\scaler.pkl"
    scaler_path = None

    main(folder, checkpoint_path, scaler_path)
