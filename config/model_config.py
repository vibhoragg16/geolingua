# Model configuration
MODEL_NAME = "gpt2"  # Start with smaller model
# MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # Upgrade later

MAX_LENGTH = 512
BATCH_SIZE = 4
LEARNING_RATE = 1e-5
NUM_EPOCHS = 10

# Geographic regions
REGIONS = {
    'us_south': 'United States South',
    'uk': 'United Kingdom',
    'australia': 'Australia',
    'india': 'India',
    'nigeria': 'Nigeria'
}

# LoRA configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

# Training configuration
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 2
SAVE_STEPS = 500
EVAL_STEPS = 250
LOGGING_STEPS = 50

# Model paths
MODEL_SAVE_PATH = "models/checkpoints"
ADAPTER_SAVE_PATH = "models/adapters"
FINAL_MODEL_PATH = "models/final"

# Device configuration
DEVICE = "cuda"  # or "cpu"
FP16 = False
DATALOADER_NUM_WORKERS = 4

# Generation parameters
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 50
DO_SAMPLE = True
MAX_NEW_TOKENS = 128
REPETITION_PENALTY = 1.1
