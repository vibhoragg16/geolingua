# You can run this in a Python script or notebook
from src.data.loaders import DataLoader
from src.data.preprocessors import DatasetPreprocessor

# Load your processed data
data_loader = DataLoader()
processed_data = data_loader.load_processed_data('processed_dataset.json')

# Split into train/val/test
train_data, val_data, test_data = data_loader.split_data(processed_data)

# Save splits
import json
with open('data/datasets/train.json', 'w') as f:
    json.dump(train_data, f)
with open('data/datasets/val.json', 'w') as f:
    json.dump(val_data, f)
with open('data/datasets/test.json', 'w') as f:
    json.dump(test_data, f)