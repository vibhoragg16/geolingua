# 🚀 GeoLingua Kaggle Training Guide

## 📋 **Complete Kaggle Procedure**

### **Step 1: Prepare Your Data** ✅

Your data is already prepared:
- ✅ `processed_dataset.json` (4.2MB) - Complete dataset
- ✅ `train_split.json` (3.1MB) - Training data (70%)
- ✅ `val_split.json` (798KB) - Validation data (15%)
- ✅ `test_split.json` (328KB) - Test data (15%)

### **Step 2: Upload Data to Kaggle**

1. **Go to Kaggle.com** → **Datasets** → **Create Dataset**
2. **Upload your data:**
   - Upload `processed_dataset.json`
   - Title: `GeoLingua Geographic Language Model Data`
   - Description: `Geographic language model training data with regional text samples from Reddit, news, and Wikipedia`
   - Tags: `nlp`, `language-model`, `geographic`, `machine-learning`
   - License: Choose appropriate license
3. **Publish the dataset**

### **Step 3: Create Kaggle Notebook**

1. **Go to Kaggle.com** → **Notebooks** → **Create Notebook**
2. **Choose settings:**
   - Language: `Python`
   - Accelerator: `GPU P100` (recommended)
   - Internet: `On`
3. **Add your dataset:**
   - Click `Add data` → `Your datasets`
   - Select your GeoLingua dataset

### **Step 4: Upload Code Files**

**Option A: Upload Individual Files**
1. Upload these files to your notebook:
   - `kaggle_train.py`
   - `kaggle_setup.py`
   - `evaluate_model.py`
   - `geolingua_training.ipynb`
   - `requirements_kaggle.txt`
   - `config/data_config.py`
   - `config/model_config.py`
   - `src/models/basemodel.py`
   - `src/models/grpo_trainer.py`
   - `src/data/loaders.py`
   - `src/data/preprocessors.py`
   - `src/utils/helpers.py`

**Option B: Use the Package**
1. Upload `geolingua_kaggle_package.zip`
2. Extract it in the notebook

### **Step 5: Run Training**

**Method 1: Using Script (Recommended)**
```python
# Install requirements
!pip install -r requirements_kaggle.txt

# Run setup
!python kaggle_setup.py

# Train model
!python kaggle_train.py
```

**Method 2: Using Notebook**
1. Open `geolingua_training.ipynb`
2. Run all cells sequentially
3. Monitor training progress

### **Step 6: Monitor Training**

**Expected Output:**
```
GeoLingua training on Kaggle...
Loaded 1605 training examples
Data distribution by region:
  australia: 409 examples
  india: 369 examples
  nigeria: 95 examples
  uk: 411 examples
  us_south: 321 examples

Splitting data with stratification by region...
Train: 1121 examples (69.8%)
Validation: 239 examples (14.9%)
Test: 245 examples (15.3%)

Initializing model: meta-llama/Llama-2-7b-chat-hf
Model on device: cuda
Trainable parameters: 4,194,304

Starting GRPO training...
Training completed! Best model saved at: /kaggle/working/models/checkpoints/best_model.pth
```

### **Step 7: Download Results**

**Files to Download:**
1. **Trained Model:** `geolingua_model.pth`
2. **Evaluation Results:** `evaluation_results.json`
3. **Data Splits:** `train_split.json`, `val_split.json`, `test_split.json`
4. **Training Logs:** Check console output

### **Step 8: Verify Results**

**Expected Evaluation Results:**
```
============================================================
TEST SET EVALUATION RESULTS
============================================================
Total test examples: 245
Overall average loss: 0.XXXX

Results by region:
----------------------------------------
australia   :  62 examples, avg loss: 0.XXXX
india       :  56 examples, avg loss: 0.XXXX
nigeria     :  15 examples, avg loss: 0.XXXX
uk          :  63 examples, avg loss: 0.XXXX
us_south    :  49 examples, avg loss: 0.XXXX
============================================================
```

## 🔧 **Troubleshooting**

### **Common Issues:**

1. **Out of Memory:**
   - Reduce batch size in `config/model_config.py`
   - Use smaller model variant
   - Enable gradient checkpointing

2. **Import Errors:**
   - Ensure all files are uploaded
   - Check file paths in Kaggle
   - Verify requirements installation

3. **Training Stuck:**
   - Check GPU usage
   - Monitor memory usage
   - Restart kernel if needed

4. **Data Loading Issues:**
   - Verify dataset path: `/kaggle/input/geolingua-data/`
   - Check file permissions
   - Ensure data format is correct

### **Performance Tips:**

1. **GPU Optimization:**
   - Use mixed precision training
   - Enable gradient accumulation
   - Monitor GPU memory usage

2. **Memory Management:**
   - Clear cache between cells
   - Use smaller batch sizes
   - Enable gradient checkpointing

3. **Time Management:**
   - Save checkpoints frequently
   - Use early stopping
   - Monitor training progress

## 📊 **Expected Training Time**

- **Setup:** 5-10 minutes
- **Data Loading:** 1-2 minutes
- **Model Initialization:** 2-3 minutes
- **Training (3 epochs):** 2-4 hours
- **Evaluation:** 5-10 minutes

**Total:** ~3-5 hours on Kaggle GPU

## 🎯 **Success Criteria**

✅ **Training Completes Successfully**
- No errors in training loop
- Loss decreases over time
- Model saves correctly

✅ **Evaluation Results**
- Test loss is reasonable
- All regions show similar performance
- No major overfitting

✅ **Files Generated**
- `geolingua_model.pth` (trained model)
- `evaluation_results.json` (metrics)
- Training logs and checkpoints

## 🚀 **Next Steps After Training**

1. **Download the trained model**
2. **Test locally** with sample inputs
3. **Deploy** for inference
4. **Share results** and insights
5. **Iterate** and improve

---

**Good luck with your GeoLingua training on Kaggle! 🎉** 