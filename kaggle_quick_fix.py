#!/usr/bin/env python3
"""
Quick fix script for Kaggle - run this to fix common issues without re-uploading the package.
"""

import os
import sys

def fix_data_paths():
    """Fix data paths for Kaggle environment."""
    
    # Fix kaggle_train.py
    if os.path.exists('kaggle_train.py'):
        with open('kaggle_train.py', 'r') as f:
            content = f.read()
        
        # Replace data paths
        content = content.replace(
            'data/processed/processed_dataset.json',
            '/kaggle/input/geolingua-geographic-language-model-data/processed_dataset.json'
        )
        content = content.replace(
            'output_dir: str = "data/processed"',
            'output_dir: str = "/kaggle/working/data/processed"'
        )
        
        with open('kaggle_train.py', 'w') as f:
            f.write(content)
        
        print("✅ Fixed data paths in kaggle_train.py")

def fix_imports():
    """Fix import issues."""
    
    # Add src to path
    if not os.path.exists('src'):
        print("⚠️  src directory not found, creating...")
        os.makedirs('src', exist_ok=True)
    
    # Create __init__.py files if missing
    init_files = [
        'src/__init__.py',
        'src/models/__init__.py',
        'src/data/__init__.py',
        'config/__init__.py'
    ]
    
    for init_file in init_files:
        if not os.path.exists(init_file):
            os.makedirs(os.path.dirname(init_file), exist_ok=True)
            with open(init_file, 'w') as f:
                f.write('# Auto-generated init file\n')
            print(f"✅ Created {init_file}")

def check_data_availability():
    """Check if data is available."""
    
    data_paths = [
        '/kaggle/input/geolingua-geographic-language-model-data/processed_dataset.json',
        'data/processed/processed_dataset.json'
    ]
    
    for path in data_paths:
        if os.path.exists(path):
            print(f"✅ Found data at: {path}")
            return path
    
    print("❌ No data found. Please upload your dataset to Kaggle.")
    return None

def create_test_script():
    """Create a simple test script to verify everything works."""
    
    test_script = '''
import sys
import os

# Add paths
sys.path.insert(0, '/kaggle/working')
sys.path.insert(0, '/kaggle/working/src')
sys.path.insert(0, '/kaggle/working/config')

try:
    from models.basemodel import GeoLinguaModel
    print("✅ GeoLinguaModel imported successfully")
except Exception as e:
    print(f"❌ Error importing GeoLinguaModel: {e}")

try:
    from models.grpo_trainer import GRPOTrainer
    print("✅ GRPOTrainer imported successfully")
except Exception as e:
    print(f"❌ Error importing GRPOTrainer: {e}")

try:
    from data.loaders import DataLoader
    print("✅ DataLoader imported successfully")
except Exception as e:
    print(f"❌ Error importing DataLoader: {e}")

# Check data
data_path = check_data_availability()
if data_path:
    print(f"✅ Data available at: {data_path}")
else:
    print("❌ No data found")
'''
    
    with open('test_imports.py', 'w') as f:
        f.write(test_script)
    
    print("✅ Created test_imports.py")

def main():
    """Run all fixes."""
    print("🔧 Running Kaggle quick fixes...")
    
    # Fix imports
    fix_imports()
    
    # Fix data paths
    fix_data_paths()
    
    # Check data
    data_path = check_data_availability()
    
    # Create test script
    create_test_script()
    
    print("\n🎉 Quick fixes completed!")
    print("\nNext steps:")
    print("1. Run: python test_imports.py")
    print("2. If imports work, run: python kaggle_train.py")
    print("3. If data not found, upload your dataset to Kaggle")

if __name__ == "__main__":
    main() 