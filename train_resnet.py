from sklearn.metrics import precision_recall_fscore_support
from torch.utils.tensorboard import SummaryWriter
from dataset import FinanceImageDataset
from torch.utils.data import DataLoader
from test import trading
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import config
import os
from model.focal_loss import FocalLoss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load pretrained ResNet and adapt for our case
resnet = models.resnet18(pretrained=True)
resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Keep original
resnet.fc = nn.Linear(resnet.fc.in_features, 3)  # 3 classes
model = resnet.to(device)

class_weights = torch.tensor([1, 2, 2]).float().to(device)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

writer = SummaryWriter('records/' + config.run_name)

min_eval_loss = float('inf')
train_dataset = FinanceImageDataset(config.train_dir, config.indicators)
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
        labels_argmax = torch.argmax(labels, dim=-1)

        # Expand to match ResNet input [B, 3, 224, 224]
        images = F.interpolate(images, size=(224, 224), mode='bilinear')
        images = images.repeat(1, 3, 1, 1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels_argmax)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        batch_count += 1

        all_predictions.extend(torch.argmax(outputs, dim=-1).cpu().numpy().flatten())
        all_labels.extend(labels_argmax.cpu().numpy().flatten())

    avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
    print(f"Average Training Loss for epoch: {epoch}: {avg_loss:.7f}", end="\t\t")
    writer.add_scalar('Training/Loss', avg_loss, epoch)

    model.eval()
    trading_result = {key: 0 for key in [
        'Final Value', 'Buy & Hold', 'Annualized Return (%)', 'Sharpe Ratio (daily)',
        'Trades Performed', 'Trades Won', 'Percent of Success', 'Avg Profit per Trade (%)',
        'Avg Trade Length (days)', 'Max Profit/Trade (%)', 'Max Loss/Trade (%)', 'Idle Ratio (%)',
        'Precision', 'Recall', 'F1']}

    total_test_loss = 0.0
    test_files = [os.path.join(config.test_dir, f) for f in os.listdir(config.test_dir)]

    for tf in test_files:
        test_dataset = FinanceImageDataset(tf, config.indicators)
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
                labels_argmax = torch.argmax(labels, dim=-1)

                images = F.interpolate(images, size=(224, 224), mode='bilinear')
                images = images.repeat(1, 3, 1, 1)

                outputs = model(images)
                loss = criterion(outputs, labels_argmax)
                test_loss += loss.item()

                all_predictions.extend(torch.argmax(outputs, dim=-1).cpu().numpy().flatten())
                all_labels.extend(labels_argmax.cpu().numpy().flatten())
                all_timestamps.extend(ts.flatten().cpu().numpy())
                all_close_candles.extend(closes.flatten().cpu().numpy())
                batch_count += 1

        file_loss = test_loss / batch_count if batch_count > 0 else 0
        total_test_loss += file_loss / len(test_files)

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='macro', zero_division=0
        )
        result, _ = trading(all_timestamps, all_close_candles, all_predictions)

        for key in result:
            trading_result[key] += result[key] / len(test_files)

        trading_result['Precision'] += precision / len(test_files)
        trading_result['Recall'] += recall / len(test_files)
        trading_result['F1'] += f1 / len(test_files)

    print(f"Test Loss : {total_test_loss:.7f}")
    writer.add_scalar('Test/Loss', total_test_loss, epoch)

    for key, item in trading_result.items():
        writer.add_scalar(f'Test/{key}', item, epoch)

    if total_test_loss < min_eval_loss:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_test_loss,
        }, os.path.join(config.checkpoint_dir, f'{config.run_name}_checkpoint_epoch_{epoch}.pth'))
        # min_eval_loss = total_test_loss

writer.close()
