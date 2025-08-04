"""
Configuration file for YOLOv8 Object Detection Pipeline
Contains all configurable parameters and settings based on project decisions.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Get the project root directory (assuming config.py is in src/)
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Input and output directories
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
CROPPED_IMAGES_DIR = OUTPUT_DIR / "cropped_images"
METADATA_DIR = OUTPUT_DIR / "metadata"

# ============================================================================
# YOLOV8 MODEL CONFIGURATION
# ============================================================================

# Model selection (Decision #1: YOLOv8n)
YOLO_MODEL_NAME = "yolov8n.pt"
YOLO_MODEL_PATH = MODELS_DIR / YOLO_MODEL_NAME

# COCO dataset classes (Decision #2: COCO dataset)
# Full list of 80 COCO classes for reference
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

# ============================================================================
# DETECTION CONFIGURATION
# ============================================================================

# ============================================================================
# DETECTION CONFIGURATION
# ============================================================================

# Quality filters for SerpAPI (to avoid wasting API calls)
QUALITY_FILTERS = {
    "min_detection_confidence": 0.35,  # Lower than before since we're not using classification
    "min_crop_width": 40,
    "min_crop_height": 40,
    "max_crop_percentage": 0.8,  # Don't process crops that are >80% of image
    "min_crop_percentage": 0.01,  # Don't process crops that are <1% of image
}

# Detection mode configuration
DETECTION_MODE = "objects_only"  # Focus on detecting objects, ignore YOLO classification
YOLO_CLASSIFICATION_THRESHOLD = None  # Not used when ignoring classifications

# Legacy confidence thresholds (for backward compatibility and multi-threshold analysis)
CONFIDENCE_THRESHOLD = QUALITY_FILTERS["min_detection_confidence"]  # Default detection confidence

CONFIDENCE_THRESHOLDS = {
    "very_conservative": 0.6,
    "conservative": 0.4,
    "balanced": 0.25,
    "aggressive": 0.15,
    "very_aggressive": 0.1
}

# NMS (Non-Maximum Suppression) threshold - YOLOv8 default is usually good
NMS_THRESHOLD = 0.7

# ============================================================================
# IMAGE CROPPING CONFIGURATION
# ============================================================================

# Padding configuration (Decision #4: 10-15 pixels padding)
PADDING_PIXELS = 15  # Fixed pixel padding
PADDING_PERCENTAGE = 0.10  # 10% of bounding box dimensions

# Padding mode selection
PADDING_MODE = "pixels"  # Options: "pixels" or "percentage"

# Minimum crop size (to avoid tiny crops)
MIN_CROP_WIDTH = 50
MIN_CROP_HEIGHT = 50

# ============================================================================
# SERPAPI CONFIGURATION
# ============================================================================

# SerpAPI settings
SERPAPI_API_KEY = os.getenv('SERPAPI_API_KEY', "")  # Load from .env file
SERPAPI_ENGINE = "google_lens"  # Primary engine for product identification
SERPAPI_FALLBACK_ENGINE = "google_images"  # Fallback if primary fails

# SerpAPI parameters
SERPAPI_PARAMS = {
    "no_cache": False,  # Use cache for duplicate images
    "num": 10,          # Number of results to return
}

# Product classification settings
USE_YOLO_CLASSIFICATION = False  # Ignore YOLO classes, use SerpAPI only
USE_DETECTION_CONFIDENCE_ONLY = True  # Focus on detection confidence, not classification
SERPAPI_TIMEOUT = 30  # Timeout for API calls in seconds

# Product data extraction
EXTRACT_PRODUCT_DETAILS = {
    "brand": True,
    "model": True,
    "price": True,
    "availability": True,
    "description": True,
    "color": True,
    "size": True
}

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

# File naming conventions - updated for product-specific naming
CROP_FILENAME_TEMPLATE = "{product_name}_{detection_id}.jpg"  # Will use product name instead of generic class
CROP_FILENAME_FALLBACK = "unknown_object_{detection_id}.jpg"  # If SerpAPI fails
METADATA_FILENAME_TEMPLATE = "{original_filename}_metadata.json"

# Image output format
OUTPUT_IMAGE_FORMAT = "JPEG"
OUTPUT_IMAGE_QUALITY = 95  # JPEG quality (1-100)

# ============================================================================
# METADATA CONFIGURATION
# ============================================================================

# Metadata fields to save (Decision #6) - enhanced for product data
METADATA_FIELDS = {
    "detection_confidence_score": True,  # YOLO detection confidence (not classification)
    "bounding_box_coordinates": True,
    "original_image_path": True,
    "cropped_image_path": True,
    "detection_timestamp": True,
    "yolo_model_used": True,
    # New SerpAPI product fields
    "product_name": True,
    "product_brand": True,
    "product_model": True,
    "product_description": True,
    "product_price": True,
    "product_color": True,
    "product_size": True,
    "serpapi_confidence": True,
    "serpapi_engine_used": True,
    "serpapi_processing_time": True,
    "visual_matches": True,  # Store multiple product matches
}

# Additional metadata options
SAVE_DETAILED_METADATA = True  # Include extra info like image dimensions
METADATA_INDENT = 2  # JSON formatting indent

# ============================================================================
# PROCESSING CONFIGURATION
# ============================================================================

# Single image processing (Decision #8)
BATCH_PROCESSING_ENABLED = False

# Error handling (Decision #7)
SAVE_EMPTY_RESULTS = True  # Save JSON even when no objects detected
SKIP_CORRUPTED_IMAGES = True

# ============================================================================
# SUPPORTED FILE FORMATS
# ============================================================================

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
ENABLE_CONSOLE_LOGGING = True
ENABLE_FILE_LOGGING = False  # Set to True to save logs to file

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_paths():
    """Create necessary directories if they don't exist."""
    directories = [
        INPUT_DIR,
        OUTPUT_DIR,
        CROPPED_IMAGES_DIR,
        METADATA_DIR,
        MODELS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ Directory ready: {directory}")

def get_config_summary():
    """Return a summary of current configuration."""
    return {
        "model": YOLO_MODEL_NAME,
        "detection_mode": DETECTION_MODE,
        "use_yolo_classification": USE_YOLO_CLASSIFICATION,
        "serpapi_engine": SERPAPI_ENGINE,
        "min_detection_confidence": QUALITY_FILTERS["min_detection_confidence"],
        "padding_mode": PADDING_MODE,
        "padding_value": PADDING_PIXELS if PADDING_MODE == "pixels" else PADDING_PERCENTAGE,
        "output_format": OUTPUT_IMAGE_FORMAT,
        "batch_processing": BATCH_PROCESSING_ENABLED,
        "product_metadata_fields": [k for k, v in METADATA_FIELDS.items() if v and k.startswith("product_")]
    }

def validate_config():
    """Validate configuration settings."""
    errors = []
    
    # Check SerpAPI configuration
    if not SERPAPI_API_KEY and not os.getenv('SERPAPI_API_KEY'):
        errors.append("SERPAPI_API_KEY must be set in config or environment variable")
    
    # Check detection confidence threshold
    min_confidence = QUALITY_FILTERS["min_detection_confidence"]
    if not 0.0 <= min_confidence <= 1.0:
        errors.append("min_detection_confidence must be between 0.0 and 1.0")
    
    # Check padding values
    if PADDING_MODE == "pixels" and PADDING_PIXELS < 0:
        errors.append("PADDING_PIXELS must be non-negative")
    elif PADDING_MODE == "percentage" and not 0.0 <= PADDING_PERCENTAGE <= 1.0:
        errors.append("PADDING_PERCENTAGE must be between 0.0 and 1.0")
    
    # Check minimum crop sizes
    if MIN_CROP_WIDTH <= 0 or MIN_CROP_HEIGHT <= 0:
        errors.append("Minimum crop dimensions must be positive")
    
    # Check SerpAPI engine
    valid_engines = ["google_lens", "google_images", "bing_visual_search"]
    if SERPAPI_ENGINE not in valid_engines:
        errors.append(f"SERPAPI_ENGINE must be one of: {valid_engines}")
    
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(errors))
    
    return True

def get_serpapi_key():
    """Get SerpAPI key from config or environment variable."""
    return SERPAPI_API_KEY or os.getenv('SERPAPI_API_KEY')

# ============================================================================
# INITIALIZATION
# ============================================================================

if __name__ == "__main__":
    print("YOLOv8 Detection Pipeline Configuration")
    print("=" * 50)
    
    # Validate configuration
    try:
        validate_config()
        print("✓ Configuration validation passed")
    except ValueError as e:
        print(f"✗ Configuration validation failed: {e}")
        exit(1)
    
    # Create directories
    validate_paths()
    
    # Print configuration summary
    print("\nConfiguration Summary:")
    print("-" * 30)
    config_summary = get_config_summary()
    for key, value in config_summary.items():
        print(f"{key}: {value}")
    
    print(f"\nProject paths:")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Input directory: {INPUT_DIR}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Models directory: {MODELS_DIR}")