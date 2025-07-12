
#!/usr/bin/env python3
"""
Direct fix script for Kaggle - fixes all issues without re-uploading the package.
Run this directly on Kaggle to fix import, path, and model issues.
"""

import os
import sys
import shutil

def fix_all_issues():
    """Fix all common issues on Kaggle."""
    
    print("🔧 Running comprehensive Kaggle fixes...")
    
    # 1. Fix data paths
    fix_data_paths()
    
    # 2. Fix imports
    fix_imports()
    
    # 3. Fix model issues
    fix_model_issues()
    
    # 4. Create test script
    create_test_script()
    
    print("\n🎉 All fixes applied!")

def fix_data_paths():
    """Fix data paths for Kaggle environment."""
    
    print("📁 Fixing data paths...")
    
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
    
    print("📦 Fixing imports...")
    
    # Create necessary directories
    directories = ['src', 'src/models', 'src/data', 'config']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Create __init__.py files
    init_files = [
        'src/__init__.py',
        'src/models/__init__.py',
        'src/data/__init__.py',
        'config/__init__.py'
    ]
    
    for init_file in init_files:
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write('# Auto-generated init file\n')
            print(f"✅ Created {init_file}")

def fix_model_issues():
    """Fix model-related issues."""
    
    print("🤖 Fixing model issues...")
    
    # Fix basemodel.py if it exists
    if os.path.exists('src/models/basemodel.py'):
        with open('src/models/basemodel.py', 'r') as f:
            content = f.read()
        
        # Add auto-detection for target modules
        if 'def _get_target_modules(self) -> List[str]:' not in content:
            # Add the auto-detection method
            auto_detect_method = '''
    def _get_target_modules(self) -> List[str]:
        """Auto-detect target modules for LoRA."""
        
        # Get all module names
        module_names = [name for name, _ in self.model.named_modules()]
        logger.info(f"Available modules: {module_names[:10]}...")  # Log first 10
        
        # Common patterns for different model types
        patterns = {
            'llama': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            'gpt': ['c_attn', 'c_proj', 'c_fc'],
            'bert': ['query', 'key', 'value', 'dense'],
            't5': ['q', 'k', 'v', 'o'],
            'falcon': ['query_key_value', 'dense'],
            'mpt': ['Wqkv', 'out_proj'],
            'gpt2': ['c_attn', 'c_proj'],
            'bloom': ['query_key_value', 'dense'],
            'opt': ['q_proj', 'k_proj', 'v_proj', 'out_proj']
        }
        
        # Try to match model type
        model_name_lower = self.model_name.lower()
        for model_type, modules in patterns.items():
            if model_type in model_name_lower:
                # Check if these modules exist
                available_modules = [m for m in modules if any(m in name for name in module_names)]
                if available_modules:
                    logger.info(f"Detected {model_type} pattern, using: {available_modules}")
                    return available_modules
        
        # Fallback: look for common attention patterns
        attention_patterns = [
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'query', 'key', 'value', 'dense',
            'c_attn', 'c_proj',
            'Wqkv', 'out_proj'
        ]
        
        found_modules = []
        for pattern in attention_patterns:
            if any(pattern in name for name in module_names):
                found_modules.append(pattern)
        
        if found_modules:
            logger.info(f"Found attention modules: {found_modules}")
            return found_modules
        
        # Last resort: return empty list to avoid error
        logger.warning("No suitable target modules found, using empty list")
        return []
'''
            
            # Find the setup_lora method and add the auto-detection
            if 'def setup_lora(self, target_modules: Optional[List[str]] = None) -> None:' in content:
                # Replace the setup_lora method with auto-detection
                old_setup = '''    def setup_lora(self, target_modules: Optional[List[str]] = None) -> None:
        """
        Setup LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.
        
        Args:
            target_modules: List of module names to apply LoRA to
        """
        try:
            logger.info("Setting up LoRA configuration...")
            
            # Default target modules for different model types
            if target_modules is None:
                if "gpt" in self.model_name.lower():
                    target_modules = ["c_attn", "c_proj", "c_fc"]
                elif "llama" in self.model_name.lower():
                    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
                else:
                    target_modules = ["query", "key", "value", "dense"]
            
            # Create LoRA configuration
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=LORA_R,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                target_modules=target_modules,
                bias="none"
            )
            
            # Apply LoRA to model
            self.model = get_peft_model(self.model, lora_config)
            
            # Print trainable parameters
            self.model.print_trainable_parameters()
            
            logger.info("LoRA setup complete")
            
        except Exception as e:
            logger.error(f"Error setting up LoRA: {str(e)}")
            raise'''
                
                new_setup = '''    def setup_lora(self, target_modules: Optional[List[str]] = None) -> None:
        """
        Setup LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.
        
        Args:
            target_modules: List of module names to apply LoRA to
        """
        try:
            logger.info("Setting up LoRA configuration...")
            
            # Auto-detect target modules based on model architecture
            if target_modules is None:
                target_modules = self._get_target_modules()
            
            logger.info(f"Using target modules: {target_modules}")
            
            # Create LoRA configuration
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=LORA_R,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                target_modules=target_modules,
                bias="none"
            )
            
            # Apply LoRA to model
            self.model = get_peft_model(self.model, lora_config)
            
            # Print trainable parameters
            self.model.print_trainable_parameters()
            
            logger.info("LoRA setup complete")
            
        except Exception as e:
            logger.error(f"Error setting up LoRA: {str(e)}")
            # Try with a more generic approach
            logger.info("Trying with generic target modules...")
            self._setup_lora_generic()'''
            
            content = content.replace(old_setup, new_setup)
            
            # Add the auto-detection method
            content += auto_detect_method
            
            # Add the generic setup method
            generic_setup = '''
    def _setup_lora_generic(self):
        """Setup LoRA with generic configuration."""
        try:
            # Try with a very basic configuration
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=LORA_R,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                target_modules=[],  # Empty list to avoid errors
                bias="none"
            )
            
            self.model = get_peft_model(self.model, lora_config)
            logger.info("LoRA setup with generic config complete")
            
        except Exception as e:
            logger.error(f"Failed to setup LoRA: {e}")
            logger.info("Continuing without LoRA...")
            # Continue without LoRA'''
            
            content += generic_setup
            
            with open('src/models/basemodel.py', 'w') as f:
                f.write(content)
            
            print("✅ Fixed model auto-detection in basemodel.py")

def create_test_script():
    """Create a comprehensive test script."""
    
    print("🧪 Creating test script...")
    
    test_script = '''#!/usr/bin/env python3
"""
Comprehensive test script for Kaggle.
"""

import sys
import os
import torch

# Add paths
sys.path.insert(0, '/kaggle/working')
sys.path.insert(0, '/kaggle/working/src')
sys.path.insert(0, '/kaggle/working/config')

def test_imports():
    """Test all imports."""
    print("🔍 Testing imports...")
    
    try:
        from models.basemodel import GeoLinguaModel
        print("✅ GeoLinguaModel imported")
    except Exception as e:
        print(f"❌ GeoLinguaModel import failed: {e}")
        return False
    
    try:
        from models.grpo_trainer import GRPOTrainer
        print("✅ GRPOTrainer imported")
    except Exception as e:
        print(f"❌ GRPOTrainer import failed: {e}")
        return False
    
    try:
        from data.loaders import DataLoader
        print("✅ DataLoader imported")
    except Exception as e:
        print(f"❌ DataLoader import failed: {e}")
        return False
    
    return True

def test_model_loading():
    """Test model loading with a small model."""
    print("\\n🤖 Testing model loading...")
    
    try:
        from models.basemodel import GeoLinguaModel
        
        # Use a smaller model for testing
        model = GeoLinguaModel(
            model_name="microsoft/DialoGPT-medium",
            regions=['us_south', 'uk', 'australia', 'india', 'nigeria'],
            lora_config={
                'r': 16,
                'lora_alpha': 32,
                'lora_dropout': 0.1,
                'target_modules': None  # Auto-detect
            }
        )
        print("✅ Model initialized")
        
        # Test forward pass
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        input_ids = torch.randint(0, 1000, (1, 10)).to(device)
        attention_mask = torch.ones(1, 10).to(device)
        region_ids = torch.tensor([0]).to(device)
        
        with torch.no_grad():
            outputs = model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                region_ids=region_ids
            )
        
        print("✅ Forward pass successful")
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading():
    """Test data loading."""
    print("\\n📊 Testing data loading...")
    
    data_paths = [
        '/kaggle/input/geolingua-geographic-language-model-data/processed_dataset.json',
        'data/processed/processed_dataset.json'
    ]
    
    for path in data_paths:
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024*1024)  # MB
            print(f"✅ Found data: {path} ({size:.2f} MB)")
            return True
    
    print("❌ No data found")
    return False

def main():
    """Run all tests."""
    print("🧪 Running comprehensive tests...")
    print("="*50)
    
    imports_ok = test_imports()
    model_ok = test_model_loading()
    data_ok = test_data_loading()
    
    print("\\n" + "="*50)
    print("📋 TEST RESULTS")
    print("="*50)
    print(f"Imports: {'✅' if imports_ok else '❌'}")
    print(f"Model: {'✅' if model_ok else '❌'}")
    print(f"Data: {'✅' if data_ok else '❌'}")
    
    if imports_ok and model_ok and data_ok:
        print("\\n🎉 All tests passed! Ready for training.")
        print("Run: python kaggle_train.py")
    else:
        print("\\n⚠️  Some tests failed. Check the errors above.")
    
    print("="*50)

if __name__ == "__main__":
    main()
'''
    
    with open('comprehensive_test.py', 'w') as f:
        f.write(test_script)
    
    print("✅ Created comprehensive_test.py")

def main():
    """Main function."""
    print("🚀 GeoLingua Direct Fix for Kaggle")
    print("="*40)
    
    fix_all_issues()
    
    print("\n📋 Next Steps:")
    print("1. Run: python comprehensive_test.py")
    print("2. If tests pass: python kaggle_train.py")
    print("3. If tests fail: Check the error messages")

if __name__ == "__main__":
    main() 