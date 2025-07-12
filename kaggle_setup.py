#!/usr/bin/env python3
"""
Kaggle setup script for GeoLingua project.
This script helps set up the project on Kaggle with proper paths and data handling.
"""

import os
import sys
import json
import shutil
from pathlib import Path

def setup_kaggle_environment():
    """Setup the environment for Kaggle."""
    
    # Create necessary directories
    directories = [
        'data/raw',
        'data/processed', 
        'data/datasets',
        'models/checkpoints',
        'logs',
        'outputs',
        'cache',
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Update config paths for Kaggle
    update_config_paths()
    
    print("Kaggle environment setup complete!")

def update_config_paths():
    """Update configuration paths for Kaggle environment."""
    
    # Read current config
    config_file = 'config/data_config.py'
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            content = f.read()
        
        # Update paths for Kaggle
        content = content.replace(
            'RAW_DATA_PATH = "data/raw"',
            'RAW_DATA_PATH = "/kaggle/input/geolingua-data/data/raw"'
        )
        content = content.replace(
            'PROCESSED_DATA_PATH = "data/processed"',
            'PROCESSED_DATA_PATH = "/kaggle/working/data/processed"'
        )
        content = content.replace(
            'DATASETS_PATH = "data/datasets"',
            'DATASETS_PATH = "/kaggle/working/data/datasets"'
        )
        
        # Write updated config
        with open(config_file, 'w') as f:
            f.write(content)
        
        print("Updated config paths for Kaggle")

def create_kaggle_notebook():
    """Create a Kaggle notebook template."""
    
    notebook_content = '''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# GeoLingua: Geographic Language Model Training on Kaggle\\n",
    "\\n",
    "This notebook trains a geographic language model using GRPO techniques."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install requirements\\n",
    "!pip install torch transformers datasets peft accelerate pandas numpy matplotlib seaborn wandb praw newspaper3k scikit-learn nltk tqdm python-dotenv lxml feedparser beautifulsoup4 wikipedia-api"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Setup environment\\n",
    "import sys\\n",
    "sys.path.append('/kaggle/working')\\n",
    "\\n",
    "# Run setup\\n",
    "!python kaggle_setup.py"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import project modules\\n",
    "from src.data.preprocessors import DatasetPreprocessor\\n",
    "from src.data.loaders import DataLoader\\n",
    "from src.models.basemodel import GeoLinguaModel\\n",
    "from src.models.grpo_trainer import GRPOTrainer\\n",
    "from config.data_config import *\\n",
    "from config.model_config import *"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load and preprocess data\\n",
    "data_loader = DataLoader()\\n",
    "processed_data = data_loader.load_processed_data('processed_dataset.json')\\n",
    "\\n",
    "print(f\"Loaded {len(processed_data)} training examples\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initialize model\\n",
    "model = GeoLinguaModel(\\n",
    "    model_name=MODEL_NAME,\\n",
    "    regions=['us_south', 'uk', 'australia', 'india', 'nigeria'],\\n",
    "    lora_config={\\n",
    "        'r': LORA_R,\\n",
    "        'lora_alpha': LORA_ALPHA,\\n",
    "        'lora_dropout': LORA_DROPOUT,\\n",
    "        'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj']\\n",
    "    }\\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Train model\\n",
    "trainer = GRPOTrainer(\\n",
    "    model=model,\\n",
    "    train_datasets=processed_data,\\n",
    "    eval_datasets=processed_data[:100],  # Use subset for evaluation\\n",
    "    learning_rate=LEARNING_RATE,\\n",
    "    num_epochs=NUM_EPOCHS,\\n",
    "    output_dir='/kaggle/working/models/checkpoints',\\n",
    "    use_wandb=False  # Disable wandb on Kaggle\\n",
    ")\\n",
    "\\n",
    "best_model_path = trainer.train()\\n",
    "print(f\"Training completed! Best model saved at: {best_model_path}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Save model for download\\n",
    "import shutil\\n",
    "shutil.copy(best_model_path, '/kaggle/working/geolingua_model.pth')\\n",
    "print(\"Model saved for download!\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}'''
    
    with open('geolingua_training.ipynb', 'w') as f:
        f.write(notebook_content)
    
    print("Created Kaggle notebook: geolingua_training.ipynb")

def create_requirements_kaggle():
    """Create a requirements file for Kaggle."""
    
    requirements = '''torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
peft>=0.4.0
accelerate>=0.20.0
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
wandb>=0.15.0
praw>=7.7.0
newspaper3k>=0.2.8
scikit-learn>=1.3.0
nltk>=3.8.0
tqdm
python-dotenv
lxml
feedparser
beautifulsoup4
wikipedia-api
'''
    
    with open('requirements_kaggle.txt', 'w') as f:
        f.write(requirements)
    
    print("Created requirements file: requirements_kaggle.txt")

def main():
    """Main setup function."""
    print("Setting up GeoLingua for Kaggle...")
    
    # Setup environment
    setup_kaggle_environment()
    
    # Create Kaggle notebook
    create_kaggle_notebook()
    
    # Create requirements file
    create_requirements_kaggle()
    
    print("\nKaggle setup complete!")
    print("\nNext steps:")
    print("1. Upload your processed data to Kaggle as a dataset")
    print("2. Create a new notebook and upload the generated files")
    print("3. Run the training notebook")

if __name__ == "__main__":
    main() 