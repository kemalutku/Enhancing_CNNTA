import pandas as pd
import torch
from torch.utils.data import Dataset


class FinanceImageDataset(Dataset):
    def __init__(self, csv_dir, feature_cols, index_dir=None, use_index=False):
        self.feature_cols = feature_cols
        self.label_col = 'Label'
        self.sequence_length = len(self.feature_cols)

        self.data = pd.read_csv(csv_dir)[self.sequence_length:]

        self.data_array = self.data.loc[:, self.feature_cols].values
        self.label_array = self.data.loc[:, 'Label'].values
        self.time_stamps = self.data.loc[:, 'Date'].values
        self.closes_array = self.data.loc[:, 'Close'].values
        self.data_length = len(self.data_array)

        self.num_classes = 3

        self.index_data = None
        if index_dir and use_index:
            self.index_data = pd.read_csv(index_dir)
            self.index_time_stamps = self.index_data.loc[:, 'Date'].values
            self.index_data_array = self.index_data.loc[:, self.feature_cols].values

    def __len__(self):
        return self.data_length - self.sequence_length

    def __getitem__(self, idx):
        x = self.data_array[idx:idx + self.sequence_length]
        y = self.label_array[idx + self.sequence_length]

        x = torch.tensor(x, dtype=torch.float32).reshape(-1, self.sequence_length, len(self.feature_cols))
        y = torch.tensor(y, dtype=torch.long)

        x = x.reshape(-1, self.sequence_length, len(self.feature_cols))
        y = torch.nn.functional.one_hot(y, num_classes=self.num_classes).float()

        timestamps = self.time_stamps[idx + self.sequence_length]
        timestamp_start = self.time_stamps[idx]
        closes = torch.tensor(self.closes_array[idx + self.sequence_length])

        if self.index_data is not None:
            # Find the matching index data
            index_idx = (self.index_time_stamps == timestamp_start).nonzero()[0][0]
            index_x = self.index_data_array[index_idx:index_idx + self.sequence_length]
            index_x = torch.tensor(index_x, dtype=torch.float32).reshape(-1, self.sequence_length,
                                                                         len(self.feature_cols))

            # Combine the original data and index data
            combined_x = torch.cat((x, index_x), dim=0)
        else:
            combined_x = x

        return timestamps, closes, combined_x, y


if __name__ == "__main__":
    import os
    import config
    import cv2

    csv_dir = r"C:\Users\KemalUtkuLekesiz\Documents\Kod\Tez\CnnTezV3\pythonProject\data\test\1d\AAPL_2017_2017_test.csv"
    index_dir = config.test_index_dir
    feature_cols = config.indicators

    dataset = FinanceImageDataset(csv_dir, feature_cols, index_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    for i, (_, _, images, _) in enumerate(dataloader):
        images = images.squeeze().numpy()
        original_data = images[0]
        index_data = images[1]
        original_data = cv2.resize(original_data, (224, 224))
        index_data = cv2.resize(index_data, (224, 224))

        cv2.imshow('Original Data', original_data)
        cv2.imshow('Index Data', index_data)

        key = cv2.waitKey(0)  # Wait for a key press to move to the next image
        if key == 27:  # Press 'Esc' to exit
            break

    cv2.destroyAllWindows()