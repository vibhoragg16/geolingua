#!/usr/bin/env python3
"""
Test script to verify model loading works correctly on Kaggle.
"""

import sys
import os
import torch

# Add paths
sys.path.insert(0, '/kaggle/working')
sys.path.insert(0, '/kaggle/working/src')
sys.path.insert(0, '/kaggle/working/config')

def test_model_loading():
    """Test if the model can be loaded successfully."""
    
    print("🔍 Testing model loading...")
    
    try:
        # Import the model
        from models.basemodel import GeoLinguaModel
        print("✅ GeoLinguaModel imported successfully")
        
        # Test model initialization
        print("🔄 Initializing model...")
        model = GeoLinguaModel(
            model_name="microsoft/DialoGPT-medium",  # Use a smaller model for testing
            regions=['us_south', 'uk', 'australia', 'india', 'nigeria'],
            lora_config={
                'r': 16,
                'lora_alpha': 32,
                'lora_dropout': 0.1,
                'target_modules': None  # Auto-detect
            }
        )
        print("✅ Model initialized successfully")
        
        # Test forward pass
        print("🔄 Testing forward pass...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        # Create dummy input
        input_ids = torch.randint(0, 1000, (1, 10)).to(device)
        attention_mask = torch.ones(1, 10).to(device)
        region_ids = torch.tensor([0]).to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                region_ids=region_ids
            )
        
        print("✅ Forward pass successful")
        print(f"📊 Output keys: {list(outputs.keys())}")
        
        # Check model info
        info = model.get_model_info()
        print(f"📈 Model info: {info}")
        
        print("\n🎉 All tests passed! Model is ready for training.")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading():
    """Test if data can be loaded."""
    
    print("\n🔍 Testing data loading...")
    
    data_paths = [
        '/kaggle/input/geolingua-geographic-language-model-data/processed_dataset.json',
        'data/processed/processed_dataset.json'
    ]
    
    for path in data_paths:
        if os.path.exists(path):
            print(f"✅ Found data at: {path}")
            # Check file size
            size = os.path.getsize(path) / (1024*1024)  # MB
            print(f"📊 File size: {size:.2f} MB")
            return path
    
    print("❌ No data found")
    return None

def main():
    """Run all tests."""
    print("🧪 Running GeoLingua tests...")
    print("="*50)
    
    # Test data loading
    data_path = test_data_loading()
    
    # Test model loading
    model_ok = test_model_loading()
    
    print("\n" + "="*50)
    print("📋 TEST SUMMARY")
    print("="*50)
    print(f"Data available: {'✅' if data_path else '❌'}")
    print(f"Model loading: {'✅' if model_ok else '❌'}")
    
    if data_path and model_ok:
        print("\n🎉 All tests passed! Ready for training.")
        print("Run: python kaggle_train.py")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
    
    print("="*50)

if __name__ == "__main__":
    main() 