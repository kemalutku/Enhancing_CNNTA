import os
import pandas as pd

def normalize():
    raw_folder = 'data/raw/dow30'
    normalized_folder = 'data/normalized'
    dow_file = os.path.join(normalized_folder, 'DJI.csv')

    # Read the DJI index values
    print(f"Looking for DJI file at: {dow_file}")
    if not os.path.exists(dow_file):
        raise FileNotFoundError(f"DJI.csv not found in {normalized_folder}")

    dow_data = pd.read_csv(dow_file)
    if 'Date' not in dow_data.columns or 'Close' not in dow_data.columns:
        raise ValueError("DJI.csv must contain 'Date' and 'Close' columns")

    dow_data.set_index('Date', inplace=True)

    # Recursively search for CSV files in the raw folder
    for root, _, files in os.walk(raw_folder):
        for file_name in files:
            if file_name.endswith('.csv'):
                raw_file_path = os.path.join(root, file_name)
                raw_data = pd.read_csv(raw_file_path)

                if 'Date' not in raw_data.columns or 'Close' not in raw_data.columns:
                    print(f"Skipping {file_name}: Missing 'Date' or 'Close' column")
                    continue

                raw_data.set_index('Date', inplace=True)

                # Align with DJI data and normalize all OHLCV columns
                aligned_data = raw_data.join(dow_data, how='inner', rsuffix='_DJI')
                for column in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if column in raw_data.columns:
                        aligned_data[column] = aligned_data[column] / aligned_data['Close_DJI']

                # Save the normalized data
                relative_path = os.path.relpath(raw_file_path, raw_folder)
                normalized_file_path = os.path.join(normalized_folder, relative_path)
                os.makedirs(os.path.dirname(normalized_file_path), exist_ok=True)
                aligned_data.reset_index()[['Date'] + [col for col in ['Open', 'High', 'Low', 'Close', 'Volume'] if col in aligned_data.columns]].to_csv(normalized_file_path, index=False)
                print(f"Normalized data saved to {normalized_file_path}")

if __name__ == "__main__":
    normalize()