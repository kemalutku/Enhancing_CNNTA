from sklearn.metrics import precision_recall_fscore_support
from torch.utils.tensorboard import SummaryWriter
from dataset import FinanceImageDataset
from torch.utils.data import DataLoader
from test import trading
import torch
import config
import os
from model.focal_loss import FocalLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = config.model(apply_bn=config.apply_bn, window_length=config.window_length).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

class_weights = torch.tensor([1, 1, 1]).to(device)
# criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
criterion = FocalLoss(alpha=class_weights, gamma=2.0)

writer = SummaryWriter('records/' + config.run_name)
writer.add_text("HPARAMS", str(config.hparams), 0)

min_eval_loss = 999

train_dataset = FinanceImageDataset(config.train_dir, config.indicators, index_dir=config.train_index_dir,
                                    use_index=config.use_index)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True)

print("Start train session.")
for epoch in range(config.max_epochs):
    model.train()

    epoch_loss = 0.0
    all_predictions = []
    all_labels = []
    batch_count = 0
    for _, _, images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        batch_count += 1

        all_predictions.extend(torch.argmax(outputs, dim=-1).cpu().numpy().flatten())
        all_labels.extend(torch.argmax(labels, dim=-1).cpu().numpy().flatten())

    avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
    print(f"Average Training Loss for epoch: {epoch}: {avg_loss:.7f}", end="\t\t")
    writer.add_scalar('Training/Loss', avg_loss, epoch)

    model.eval()

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

    total_test_loss = 0.0
    test_files = [os.path.join(config.test_dir, f) for f in os.listdir(config.test_dir)]
    for tf in test_files:
        test_dataset = FinanceImageDataset(tf, config.indicators, index_dir=config.test_index_dir,
                                           use_index=config.use_index)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

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

                all_predictions.extend(torch.argmax(outputs, dim=-1).cpu().numpy().flatten())
                all_labels.extend(torch.argmax(labels, dim=-1).cpu().numpy().flatten())
                all_timestamps.extend(ts.flatten().cpu().numpy())
                all_close_candles.extend(closes.flatten().cpu().numpy())
                batch_count += 1

            file_loss = test_loss / batch_count if batch_count > 0 else 0
            total_test_loss += file_loss / len(test_files)

        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro',
                                                                   zero_division=0)

        result, _ = trading(all_timestamps, all_close_candles, all_predictions)

        for key, item in result.items():
            trading_result[key] += item / len(test_files)

        trading_result['Precision'] += precision / len(test_files)
        trading_result['Recall'] += recall / len(test_files)
        trading_result['F1'] += f1 / len(test_files)

    print(f"Test Loss : {total_test_loss:.7f}")
    writer.add_scalar('Test/Loss', total_test_loss, epoch)

    for key, item in trading_result.items():
        writer.add_scalar(f'Test/{key}', item, epoch)

    if total_test_loss < min_eval_loss:
        # min_eval_loss = total_test_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_test_loss,
        }, os.path.join(config.checkpoint_dir, f'{config.run_name}_checkpoint_epoch_{epoch}.pth'))

writer.close()
