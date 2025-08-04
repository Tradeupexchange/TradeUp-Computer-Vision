"""
Utility functions for YOLOv8 Object Detection Pipeline
Contains helper functions for file operations, JSON handling, image processing, and logging.
"""

import json
import logging
import os
import shutil
import time
import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import cv2
import numpy as np
import requests
from PIL import Image

try:
    from .config import (
        SUPPORTED_IMAGE_EXTENSIONS,
        METADATA_INDENT,
        LOG_LEVEL,
        LOG_FORMAT,
        ENABLE_CONSOLE_LOGGING,
        ENABLE_FILE_LOGGING,
        OUTPUT_IMAGE_FORMAT,
        OUTPUT_IMAGE_QUALITY,
        PROJECT_ROOT,
        # SerpAPI configuration
        SERPAPI_ENGINE,
        SERPAPI_FALLBACK_ENGINE,
        SERPAPI_PARAMS,
        SERPAPI_TIMEOUT,
        EXTRACT_PRODUCT_DETAILS,
        get_serpapi_key
    )
except ImportError:
    # Handle case when running as main module
    from config import (
        SUPPORTED_IMAGE_EXTENSIONS,
        METADATA_INDENT,
        LOG_LEVEL,
        LOG_FORMAT,
        ENABLE_CONSOLE_LOGGING,
        ENABLE_FILE_LOGGING,
        OUTPUT_IMAGE_FORMAT,
        OUTPUT_IMAGE_QUALITY,
        PROJECT_ROOT,
        # SerpAPI configuration
        SERPAPI_ENGINE,
        SERPAPI_FALLBACK_ENGINE,
        SERPAPI_PARAMS,
        SERPAPI_TIMEOUT,
        EXTRACT_PRODUCT_DETAILS,
        get_serpapi_key
    )

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(name: str = "yolov8_pipeline") -> logging.Logger:
    """
    Set up logging configuration for the pipeline.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL.upper()))
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    formatter = logging.Formatter(LOG_FORMAT)
    
    # Console handler
    if ENABLE_CONSOLE_LOGGING:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if ENABLE_FILE_LOGGING:
        log_file = PROJECT_ROOT / "logs" / "pipeline.log"
        log_file.parent.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

# ============================================================================
# FILE AND PATH UTILITIES
# ============================================================================

def is_image_file(file_path: Union[str, Path]) -> bool:
    """
    Check if a file is a supported image format.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if file is a supported image format
    """
    return Path(file_path).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS

def get_safe_filename(filename: str) -> str:
    """
    Create a safe filename by removing/replacing problematic characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename for filesystem
    """
    # Replace problematic characters
    safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_"
    safe_filename = "".join(c if c in safe_chars else "_" for c in filename)
    
    # Remove multiple consecutive underscores
    while "__" in safe_filename:
        safe_filename = safe_filename.replace("__", "_")
    
    # Remove leading/trailing underscores
    safe_filename = safe_filename.strip("_")
    
    return safe_filename

def create_output_directory(image_path: Union[str, Path], base_output_dir: Path) -> Path:
    """
    Create hierarchical output directory for an image.
    
    Args:
        image_path: Path to the original image
        base_output_dir: Base output directory
        
    Returns:
        Created output directory path
    """
    image_path = Path(image_path)
    # Create directory name from image filename (without extension)
    safe_name = get_safe_filename(image_path.stem)
    output_dir = base_output_dir / f"{safe_name}_{image_path.suffix[1:]}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Created output directory: {output_dir}")
    
    return output_dir

def validate_input_image(image_path: Union[str, Path]) -> bool:
    """
    Validate that an input image exists and is readable.
    
    Args:
        image_path: Path to the image
        
    Returns:
        True if image is valid and readable
    """
    image_path = Path(image_path)
    
    # Check if file exists
    if not image_path.exists():
        logger.error(f"Image file does not exist: {image_path}")
        return False
    
    # Check if it's a supported format
    if not is_image_file(image_path):
        logger.error(f"Unsupported image format: {image_path.suffix}")
        return False
    
    # Try to open the image
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify image integrity
        logger.debug(f"Image validation passed: {image_path}")
        return True
    except Exception as e:
        logger.error(f"Image validation failed for {image_path}: {e}")
        return False

# ============================================================================
# IMAGE PROCESSING UTILITIES
# ============================================================================

def load_image(image_path: Union[str, Path]) -> Optional[np.ndarray]:
    """
    Load an image using OpenCV.
    
    Args:
        image_path: Path to the image
        
    Returns:
        Image as numpy array (BGR format) or None if failed
    """
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        
        logger.debug(f"Successfully loaded image: {image_path} - Shape: {image.shape}")
        return image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None

def get_image_dimensions(image_path: Union[str, Path]) -> Optional[Tuple[int, int]]:
    """
    Get image dimensions without fully loading the image.
    
    Args:
        image_path: Path to the image
        
    Returns:
        Tuple of (width, height) or None if failed
    """
    try:
        with Image.open(image_path) as img:
            return img.size  # PIL returns (width, height)
    except Exception as e:
        logger.error(f"Error getting image dimensions for {image_path}: {e}")
        return None

def save_image(image: np.ndarray, output_path: Union[str, Path], 
               format: str = OUTPUT_IMAGE_FORMAT, quality: int = OUTPUT_IMAGE_QUALITY) -> bool:
    """
    Save an image using PIL for better format control.
    
    Args:
        image: Image as numpy array (BGR format from OpenCV)
        output_path: Where to save the image
        format: Output format (JPEG, PNG, etc.)
        quality: JPEG quality (1-100)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Convert BGR to RGB for PIL
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Save with appropriate parameters
        save_kwargs = {}
        if format.upper() == 'JPEG':
            save_kwargs['quality'] = quality
            save_kwargs['optimize'] = True
        
        pil_image.save(output_path, format=format, **save_kwargs)
        logger.debug(f"Successfully saved image: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving image to {output_path}: {e}")
        return False

# ============================================================================
# BOUNDING BOX UTILITIES
# ============================================================================

def apply_padding_to_bbox(bbox: List[float], image_shape: Tuple[int, int, int], 
                         padding_pixels: int = 0, padding_percentage: float = 0.0) -> List[int]:
    """
    Apply padding to a bounding box while keeping it within image boundaries.
    
    Args:
        bbox: Bounding box as [x1, y1, x2, y2]
        image_shape: Image shape as (height, width, channels)
        padding_pixels: Fixed pixel padding
        padding_percentage: Percentage-based padding (0.0-1.0)
        
    Returns:
        Padded bounding box as [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = bbox
    height, width = image_shape[:2]
    
    # Calculate padding amounts
    if padding_percentage > 0:
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        pad_x = int(bbox_width * padding_percentage)
        pad_y = int(bbox_height * padding_percentage)
    else:
        pad_x = padding_pixels
        pad_y = padding_pixels
    
    # Apply padding
    x1_padded = max(0, int(x1 - pad_x))
    y1_padded = max(0, int(y1 - pad_y))
    x2_padded = min(width, int(x2 + pad_x))
    y2_padded = min(height, int(y2 + pad_y))
    
    logger.debug(f"Applied padding: original {bbox} -> padded [{x1_padded}, {y1_padded}, {x2_padded}, {y2_padded}]")
    
    return [x1_padded, y1_padded, x2_padded, y2_padded]

def is_valid_bbox(bbox: List[int], min_width: int = 10, min_height: int = 10) -> bool:
    """
    Check if a bounding box is valid (has minimum dimensions).
    
    Args:
        bbox: Bounding box as [x1, y1, x2, y2]
        min_width: Minimum width requirement
        min_height: Minimum height requirement
        
    Returns:
        True if bounding box is valid
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    
    valid = width >= min_width and height >= min_height
    if not valid:
        logger.debug(f"Invalid bbox dimensions: {width}x{height} (min: {min_width}x{min_height})")
    
    return valid

# ============================================================================
# JSON UTILITIES
# ============================================================================

def save_json(data: Dict[str, Any], output_path: Union[str, Path]) -> bool:
    """
    Save data to a JSON file with proper formatting.
    
    Args:
        data: Data to save
        output_path: Where to save the JSON file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=METADATA_INDENT, ensure_ascii=False, default=str)
        
        logger.debug(f"Successfully saved JSON: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving JSON to {output_path}: {e}")
        return False

def load_json(json_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Load data from a JSON file.
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        Loaded data or None if failed
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.debug(f"Successfully loaded JSON: {json_path}")
        return data
        
    except Exception as e:
        logger.error(f"Error loading JSON from {json_path}: {e}")
        return None

# ============================================================================
# SERPAPI UTILITIES
# ============================================================================

def encode_image_for_serpapi(image_path: Union[str, Path]) -> Optional[str]:
    """
    Encode an image to base64 for SerpAPI upload with size optimization.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded image string or None if failed
    """
    try:
        # Load image and resize if too large
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return None
        
        # Resize image if it's too large (SerpAPI works better with smaller images)
        height, width = image.shape[:2]
        max_dimension = 1024  # Max width or height
        
        if max(height, width) > max_dimension:
            if height > width:
                new_height = max_dimension
                new_width = int(width * (max_dimension / height))
            else:
                new_width = max_dimension
                new_height = int(height * (max_dimension / width))
            
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            logger.debug(f"Resized image from {width}x{height} to {new_width}x{new_height}")
        
        # Encode to JPEG with reasonable quality
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        success, encoded_image = cv2.imencode('.jpg', image, encode_param)
        
        if not success:
            logger.error(f"Failed to encode image: {image_path}")
            return None
        
        # Convert to base64
        encoded_string = base64.b64encode(encoded_image.tobytes()).decode('utf-8')
        
        logger.debug(f"Successfully encoded image: {image_path} (size: {len(encoded_string)} chars)")
        return encoded_string
        
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {e}")
        return None

def call_serpapi_visual_search(image_path: Union[str, Path], 
                              engine: str = SERPAPI_ENGINE) -> Optional[Dict[str, Any]]:
    """
    Call SerpAPI for visual search on an image using POST method.
    
    Args:
        image_path: Path to the cropped image
        engine: SerpAPI engine to use
        
    Returns:
        SerpAPI response dictionary or None if failed
    """
    api_key = get_serpapi_key()
    if not api_key:
        logger.error("SerpAPI key not found. Please set SERPAPI_API_KEY in .env file")
        return None
    
    start_time = time.time()
    
    try:
        # Encode image with size optimization
        encoded_image = encode_image_for_serpapi(image_path)
        if not encoded_image:
            return None
        
        # Prepare API request using POST method to avoid URI length limits
        url = "https://serpapi.com/search"
        
        data = {
            "engine": engine,
            "api_key": api_key,
            "encoded_image": encoded_image,
            **SERPAPI_PARAMS
        }
        
        logger.debug(f"Calling SerpAPI with engine: {engine} (POST method)")
        
        # Make POST request instead of GET to handle large images
        response = requests.post(url, data=data, timeout=SERPAPI_TIMEOUT)
        response.raise_for_status()
        
        processing_time = time.time() - start_time
        result = response.json()
        
        # Add metadata
        result["_serpapi_meta"] = {
            "processing_time": round(processing_time, 3),
            "engine_used": engine,
            "image_path": str(image_path)
        }
        
        logger.debug(f"SerpAPI call successful in {processing_time:.2f}s")
        return result
        
    except requests.exceptions.Timeout:
        logger.error(f"SerpAPI request timeout after {SERPAPI_TIMEOUT}s")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"SerpAPI request failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in SerpAPI call: {e}")
        return None

def call_serpapi_with_fallback(image_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Call SerpAPI with primary engine, fallback to secondary if failed.
    
    Args:
        image_path: Path to the cropped image
        
    Returns:
        SerpAPI response dictionary or None if all engines failed
    """
    # Try primary engine
    result = call_serpapi_visual_search(image_path, SERPAPI_ENGINE)
    
    if result and "visual_matches" in result and result["visual_matches"]:
        logger.debug(f"Primary engine {SERPAPI_ENGINE} successful")
        return result
    
    # Try fallback engine
    logger.info(f"Primary engine failed, trying fallback: {SERPAPI_FALLBACK_ENGINE}")
    fallback_result = call_serpapi_visual_search(image_path, SERPAPI_FALLBACK_ENGINE)
    
    if fallback_result:
        logger.debug(f"Fallback engine {SERPAPI_FALLBACK_ENGINE} successful")
        return fallback_result
    
    logger.warning(f"Both SerpAPI engines failed for image: {image_path}")
    return None

def extract_product_info(serpapi_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract structured product information from SerpAPI response.
    
    Args:
        serpapi_response: Raw SerpAPI response
        
    Returns:
        Structured product information dictionary
    """
    product_info = {
        "product_name": "Unknown Product",
        "brand": None,
        "model": None,
        "description": None,
        "price": None,
        "color": None,
        "size": None,
        "confidence": 0.0,
        "visual_matches": [],
        "serpapi_engine": serpapi_response.get("_serpapi_meta", {}).get("engine_used"),
        "processing_time": serpapi_response.get("_serpapi_meta", {}).get("processing_time", 0)
    }
    
    try:
        # Extract visual matches
        visual_matches = serpapi_response.get("visual_matches", [])
        
        if not visual_matches:
            logger.debug("No visual matches found in SerpAPI response")
            return product_info
        
        # Use the first (best) match for primary product info
        best_match = visual_matches[0]
        
        # Extract product name
        title = best_match.get("title", "")
        if title:
            product_info["product_name"] = clean_product_name(title)
        
        # Extract brand and model from title
        brand, model = extract_brand_and_model(title)
        product_info["brand"] = brand
        product_info["model"] = model
        
        # Extract other details
        product_info["description"] = best_match.get("snippet", "")
        product_info["price"] = extract_price_from_match(best_match)
        
        # Store all visual matches for reference
        product_info["visual_matches"] = visual_matches[:5]  # Keep top 5 matches
        
        # Calculate confidence based on number of similar matches
        product_info["confidence"] = calculate_product_confidence(visual_matches)
        
        logger.debug(f"Extracted product: {product_info['product_name']}")
        
    except Exception as e:
        logger.error(f"Error extracting product info: {e}")
    
    return product_info

def clean_product_name(title: str) -> str:
    """
    Clean and format product name for file naming.
    
    Args:
        title: Raw product title from SerpAPI
        
    Returns:
        Cleaned product name suitable for file naming
    """
    # Remove common noise words and characters
    noise_words = ["buy", "shop", "online", "free shipping", "best price", "sale"]
    
    # Clean the title
    cleaned = title.lower()
    
    # Remove noise words
    for word in noise_words:
        cleaned = cleaned.replace(word, "")
    
    # Remove extra whitespace and special characters
    cleaned = " ".join(cleaned.split())  # Normalize whitespace
    
    # Truncate if too long
    if len(cleaned) > 50:
        cleaned = cleaned[:50].strip()
    
    # Make safe for filenames
    safe_name = get_safe_filename(cleaned)
    
    return safe_name or "unknown_product"

def extract_brand_and_model(title: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract brand and model from product title.
    
    Args:
        title: Product title
        
    Returns:
        Tuple of (brand, model) or (None, None)
    """
    # Common brands to look for
    brands = [
        "apple", "samsung", "google", "microsoft", "sony", "nike", "adidas",
        "hydroflask", "yeti", "starbucks", "keurig", "airpods", "iphone",
        "macbook", "ipad", "kindle", "fitbit", "garmin", "bose", "beats"
    ]
    
    title_lower = title.lower()
    
    # Find brand
    brand = None
    for b in brands:
        if b in title_lower:
            brand = b.title()
            break
    
    # Extract model (this is simplified - could be more sophisticated)
    model = None
    words = title.split()
    
    # Look for model patterns (numbers, specific keywords)
    for i, word in enumerate(words):
        if any(char.isdigit() for char in word) and len(word) <= 10:
            # Likely a model number
            model = word
            break
    
    return brand, model

def extract_price_from_match(match: Dict[str, Any]) -> Optional[str]:
    """
    Extract price from SerpAPI visual match.
    
    Args:
        match: Single visual match from SerpAPI
        
    Returns:
        Price string or None
    """
    # Look for price in various fields
    price_fields = ["price", "displayed_price", "snippet"]
    
    for field in price_fields:
        if field in match:
            price_text = str(match[field])
            # Simple price extraction (could be more sophisticated)
            import re
            price_match = re.search(r'\$[\d,]+\.?\d*', price_text)
            if price_match:
                return price_match.group()
    
    return None

def calculate_product_confidence(visual_matches: List[Dict]) -> float:
    """
    Calculate confidence score based on visual matches.
    
    Args:
        visual_matches: List of visual matches from SerpAPI
        
    Returns:
        Confidence score between 0.0 and 1.0
    """
    if not visual_matches:
        return 0.0
    
    # Simple confidence calculation based on:
    # - Number of matches
    # - Consistency of product names
    
    num_matches = len(visual_matches)
    
    # Base confidence from number of matches
    if num_matches >= 5:
        base_confidence = 0.9
    elif num_matches >= 3:
        base_confidence = 0.7
    elif num_matches >= 2:
        base_confidence = 0.5
    else:
        base_confidence = 0.3
    
    # Check for consistency in product names
    titles = [match.get("title", "").lower() for match in visual_matches[:3]]
    
    # Simple consistency check - if titles have common words
    if titles:
        first_title_words = set(titles[0].split())
        consistency_score = sum(
            len(first_title_words.intersection(set(title.split()))) / max(len(first_title_words), 1)
            for title in titles[1:]
        ) / max(len(titles) - 1, 1)
        
        # Boost confidence if titles are consistent
        base_confidence += consistency_score * 0.2
    
    return min(base_confidence, 1.0)

def create_product_filename(product_info: Dict[str, Any], detection_id: int) -> str:
    """
    Create a filename based on product information.
    
    Args:
        product_info: Product information from SerpAPI
        detection_id: Detection ID for uniqueness
        
    Returns:
        Filename for the cropped image
    """
    try:
        product_name = product_info.get("product_name", "unknown_product")
        
        # Clean and format for filename
        safe_name = get_safe_filename(product_name)
        
        # Ensure it's not too long
        if len(safe_name) > 40:
            safe_name = safe_name[:40]
        
        filename = f"{safe_name}_{detection_id}.jpg"
        
        logger.debug(f"Generated filename: {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"Error creating product filename: {e}")
        return f"unknown_object_{detection_id}.jpg"

def create_enhanced_detection_metadata(original_image_path: Union[str, Path], 
                                     detections: List[Dict], 
                                     yolo_model_name: str,
                                     detection_time: float = 0.0,
                                     total_serpapi_time: float = 0.0) -> Dict[str, Any]:
    """
    Create comprehensive metadata for detection results with product information.
    
    Args:
        original_image_path: Path to the original image
        detections: List of detection dictionaries with product info
        yolo_model_name: Name of the YOLO model used
        detection_time: Time taken for YOLO detection
        total_serpapi_time: Total time for all SerpAPI calls
        
    Returns:
        Complete metadata dictionary with product information
    """
    original_image_path = Path(original_image_path)
    image_dimensions = get_image_dimensions(original_image_path)
    
    # Count products by brand
    brands_detected = {}
    total_serpapi_calls = 0
    successful_classifications = 0
    
    for detection in detections:
        if "product_info" in detection:
            total_serpapi_calls += 1
            brand = detection["product_info"].get("brand")
            if brand and brand != "None":
                successful_classifications += 1
                brands_detected[brand] = brands_detected.get(brand, 0) + 1
    
    metadata = {
        "original_image": {
            "path": str(original_image_path),
            "filename": original_image_path.name,
            "dimensions": {
                "width": image_dimensions[0] if image_dimensions else None,
                "height": image_dimensions[1] if image_dimensions else None
            }
        },
        "processing_info": {
            "timestamp": datetime.now().isoformat(),
            "yolo_model_used": yolo_model_name,
            "detection_time_seconds": round(detection_time, 3),
            "serpapi_time_seconds": round(total_serpapi_time, 3),
            "total_processing_time": round(detection_time + total_serpapi_time, 3),
            "total_detections": len(detections),
            "serpapi_calls_made": total_serpapi_calls,
            "successful_classifications": successful_classifications,
            "classification_success_rate": round(successful_classifications / max(total_serpapi_calls, 1) * 100, 1)
        },
        "product_summary": {
            "brands_detected": brands_detected,
            "unique_brands": len(brands_detected),
            "products_identified": successful_classifications
        },
        "detections": detections
    }
    
    return metadata

def create_empty_enhanced_metadata(original_image_path: Union[str, Path], 
                                 yolo_model_name: str,
                                 detection_time: float = 0.0) -> Dict[str, Any]:
    """
    Create metadata for images with no detections.
    
    Args:
        original_image_path: Path to the original image
        yolo_model_name: Name of the YOLO model used
        detection_time: Time taken for detection
        
    Returns:
        Metadata dictionary with empty detections
    """
    return create_enhanced_detection_metadata(original_image_path, [], yolo_model_name, detection_time, 0.0)

# ============================================================================
# GENERAL UTILITIES
# ============================================================================

def format_file_size(size_bytes: int) -> str:
    """
    Convert file size to human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get comprehensive file information.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with file information
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return {"error": "File does not exist"}
    
    stat = file_path.stat()
    
    return {
        "path": str(file_path),
        "name": file_path.name,
        "size_bytes": stat.st_size,
        "size_formatted": format_file_size(stat.st_size),
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "is_image": is_image_file(file_path)
    }

def print_enhanced_processing_summary(total_images: int, successful: int, failed: int, 
                                    total_detections: int, total_products_identified: int,
                                    total_processing_time: float, total_serpapi_time: float,
                                    serpapi_calls_made: int):
    """
    Print an enhanced summary of processing results including SerpAPI stats.
    
    Args:
        total_images: Total number of images processed
        successful: Number of successfully processed images
        failed: Number of failed images
        total_detections: Total number of objects detected
        total_products_identified: Number of products successfully identified
        total_processing_time: Total processing time
        total_serpapi_time: Total SerpAPI time
        serpapi_calls_made: Number of SerpAPI calls made
    """
    print("\n" + "="*60)
    print("ENHANCED PROCESSING SUMMARY")
    print("="*60)
    print(f"Images processed: {total_images} (✓ {successful}, ✗ {failed})")
    print(f"Objects detected: {total_detections}")
    print(f"Products identified: {total_products_identified}/{total_detections}")
    print(f"Identification success rate: {(total_products_identified/max(total_detections, 1)*100):.1f}%")
    print("-" * 60)
    print(f"YOLO detection time: {(total_processing_time - total_serpapi_time):.2f}s")
    print(f"SerpAPI classification time: {total_serpapi_time:.2f}s")
    print(f"Total processing time: {total_processing_time:.2f}s")
    print(f"Average time per image: {total_processing_time/max(total_images, 1):.2f}s")
    print("-" * 60)
    print(f"SerpAPI calls made: {serpapi_calls_made}")
    print(f"Average SerpAPI time per call: {(total_serpapi_time/max(serpapi_calls_made, 1)):.2f}s")
    if serpapi_calls_made > 0:
        estimated_cost = serpapi_calls_made * 0.01  # Rough estimate
        print(f"Estimated SerpAPI cost: ~${estimated_cost:.2f}")
    print("="*60)

# ============================================================================
# TESTING UTILITIES
# ============================================================================

def test_utilities():
    """Test utility functions including SerpAPI connectivity."""
    logger.info("Testing utility functions...")
    
    # Test filename safety
    unsafe_name = "test file!@#$%^&*()name.jpg"
    safe_name = get_safe_filename(unsafe_name)
    logger.info(f"Filename safety test: '{unsafe_name}' -> '{safe_name}'")
    
    # Test product name cleaning
    messy_product = "Apple AirPods Pro (3rd Generation) - Buy Online - Free Shipping!"
    clean_product = clean_product_name(messy_product)
    logger.info(f"Product name cleaning test: '{messy_product}' -> '{clean_product}'")
    
    # Test brand/model extraction
    brand, model = extract_brand_and_model("Apple iPhone 15 Pro 128GB")
    logger.info(f"Brand/model extraction test: brand='{brand}', model='{model}'")
    
    # Test SerpAPI key availability
    api_key = get_serpapi_key()
    if api_key:
        logger.info("✓ SerpAPI key found")
    else:
        logger.warning("✗ SerpAPI key not found - check .env file")
    
    # Test JSON operations
    test_data = {"test": "data", "number": 42, "timestamp": datetime.now()}
    temp_json = Path("temp_test.json")
    
    if save_json(test_data, temp_json):
        loaded_data = load_json(temp_json)
        logger.info(f"JSON test successful: {loaded_data is not None}")
        temp_json.unlink()  # Delete test file
    
    logger.info("Utility function tests completed")

if __name__ == "__main__":
    test_utilities()