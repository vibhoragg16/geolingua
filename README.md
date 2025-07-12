# GeoLingua: Geographic Language Model

## Quick Start

1. **Install requirements:**
   ```bash
   pip install -r requirements_kaggle.txt
   ```

2. **Run quick fix (if needed):**
   ```bash
   python kaggle_quick_fix.py
   ```

3. **Run setup:**
   ```bash
   python kaggle_setup.py
   ```

4. **Train model:**
   ```bash
   python kaggle_train.py
   ```

5. **Or use the notebook:**
   - Open `geolingua_training.ipynb`
   - Run all cells

## Troubleshooting

If you encounter import or path issues:
1. Run `python kaggle_quick_fix.py` to fix common issues
2. Run `python test_imports.py` to verify everything works
3. Check that your dataset is properly added to the notebook

## Data Structure

- `processed_dataset.json`: Complete processed dataset
- `train_split.json`: Training data (70%)
- `val_split.json`: Validation data (15%) 
- `test_split.json`: Test data (15%)

## Model Architecture

- Base model: Llama-2-7b-chat-hf
- LoRA adaptation for geographic regions
- GRPO training technique
- Regions: US South, UK, Australia, India, Nigeria

## Output

- Trained model saved as `geolingua_model.pth`
- Evaluation results in `evaluation_results.json`
