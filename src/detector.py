"""
YOLOv8 Object Detection Module - SerpAPI Optimized
Handles model loading, inference, and result processing for SerpAPI integration pipeline.
"""

import time
import base64
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

try:
    from .config import (
        YOLO_MODEL_NAME,
        YOLO_MODEL_PATH,
        NMS_THRESHOLD,
        USE_YOLO_CLASSIFICATION,
        USE_DETECTION_CONFIDENCE_ONLY,
        DETECTION_MODE,
        QUALITY_FILTERS,
        METADATA_FIELDS
    )
    from .utils import (
        setup_logging,
        load_image,
        validate_input_image,
        create_enhanced_detection_metadata,
        create_empty_enhanced_metadata
    )
except ImportError:
    # Handle case when running as main module
    from config import (
        YOLO_MODEL_NAME,
        YOLO_MODEL_PATH,
        NMS_THRESHOLD,
        USE_YOLO_CLASSIFICATION,
        USE_DETECTION_CONFIDENCE_ONLY,
        DETECTION_MODE,
        QUALITY_FILTERS,
        METADATA_FIELDS
    )
    from utils import (
        setup_logging,
        load_image,
        validate_input_image,
        create_enhanced_detection_metadata,
        create_empty_enhanced_metadata
    )

# Initialize logger
logger = setup_logging("detector")

class YOLOv8Detector:
    """
    YOLOv8 Object Detection class optimized for SerpAPI integration.
    Focuses on object detection confidence rather than classification accuracy.
    """
    
    def __init__(self, model_path: Optional[Union[str, Path]] = None, 
                 detection_confidence: float = QUALITY_FILTERS["min_detection_confidence"]):
        """
        Initialize the YOLOv8 detector for SerpAPI pipeline.
        
        Args:
            model_path: Path to YOLOv8 model file (if None, uses config default)
            detection_confidence: Detection confidence threshold (not classification)
        """
        self.model_path = Path(model_path) if model_path else YOLO_MODEL_PATH
        self.detection_confidence = detection_confidence
        self.model = None
        self.model_loaded = False
        
        # SerpAPI image processing settings
        self.max_image_size = (1024, 1024)  # Max dimensions for SerpAPI
        self.max_file_size_mb = 1.5  # Max file size in MB for SerpAPI
        self.jpeg_quality = 85  # JPEG quality for compression
        
        logger.info(f"Initializing YOLOv8 detector with model: {self.model_path}")
        logger.info(f"Detection mode: {DETECTION_MODE}")
        logger.info(f"Detection confidence threshold: {self.detection_confidence}")
        logger.info(f"Using YOLO classification: {USE_YOLO_CLASSIFICATION}")
        
        # Load model on initialization
        self.load_model()
    
    def load_model(self) -> bool:
        """
        Load the YOLOv8 model.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            # Check if model file exists, if not YOLO will download it
            if not self.model_path.exists():
                logger.info(f"Model file not found at {self.model_path}")
                logger.info("YOLO will download the model automatically...")
                
                # Create models directory if it doesn't exist
                self.model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Load the model
            logger.info("Loading YOLOv8 model...")
            self.model = YOLO(str(self.model_path))
            
            # Verify model loaded correctly
            if self.model is None:
                raise Exception("Model failed to load")
            
            self.model_loaded = True
            logger.info(f"Successfully loaded YOLOv8 model: {YOLO_MODEL_NAME}")
            
            # Log model information
            self._log_model_info()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            self.model_loaded = False
            return False
    
    def _log_model_info(self):
        """Log information about the loaded model."""
        if self.model is not None:
            try:
                # Get model info if available
                num_classes = len(self.model.names) if hasattr(self.model, 'names') else 0
                
                model_info = {
                    "model_name": YOLO_MODEL_NAME,
                    "detection_confidence_threshold": self.detection_confidence,
                    "nms_threshold": NMS_THRESHOLD,
                    "num_classes": num_classes,
                    "detection_mode": DETECTION_MODE,
                    "use_yolo_classification": USE_YOLO_CLASSIFICATION
                }
                
                logger.info("Model configuration:")
                for key, value in model_info.items():
                    logger.info(f"  {key}: {value}")
                    
            except Exception as e:
                logger.warning(f"Could not retrieve model info: {e}")
    
    def prepare_image_for_serpapi(self, image_path: Union[str, Path]) -> Optional[str]:
        """
        Prepare an image for SerpAPI by resizing and encoding to base64.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded image string, or None if failed
        """
        try:
            image_path = Path(image_path)
            
            # Log original image info
            original_size = image_path.stat().st_size
            logger.debug(f"Original image size: {original_size / 1024:.1f} KB")
            
            # Open and process image
            with Image.open(image_path) as img:
                # Log original dimensions
                logger.debug(f"Original dimensions: {img.size}")
                
                # Convert to RGB if needed (removes alpha channel)
                if img.mode in ('RGBA', 'P', 'LA'):
                    logger.debug(f"Converting from {img.mode} to RGB")
                    img = img.convert('RGB')
                
                # Resize if too large
                if img.size[0] > self.max_image_size[0] or img.size[1] > self.max_image_size[1]:
                    logger.debug(f"Resizing image from {img.size} to fit {self.max_image_size}")
                    img.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)
                    logger.debug(f"New dimensions: {img.size}")
                
                # Convert to bytes with initial quality
                img_bytes = self._image_to_bytes(img, self.jpeg_quality)
                
                # Reduce quality if still too large
                quality = self.jpeg_quality
                max_bytes = int(self.max_file_size_mb * 1024 * 1024)
                
                while len(img_bytes) > max_bytes and quality > 30:
                    quality -= 10
                    logger.debug(f"Reducing quality to {quality}% (size: {len(img_bytes) / 1024:.1f} KB)")
                    img_bytes = self._image_to_bytes(img, quality)
                
                if len(img_bytes) > max_bytes:
                    logger.warning(f"Could not reduce image size below {self.max_file_size_mb}MB limit")
                    return None
                
                # Encode to base64
                base64_string = base64.b64encode(img_bytes).decode('utf-8')
                
                logger.info(f"Image prepared for SerpAPI: {len(img_bytes) / 1024:.1f} KB, quality: {quality}%")
                return base64_string
                
        except Exception as e:
            logger.error(f"Failed to prepare image for SerpAPI: {e}")
            return None
    
    def _image_to_bytes(self, img: Image.Image, quality: int) -> bytes:
        """Convert PIL Image to bytes with specified JPEG quality."""
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=quality, optimize=True)
        return img_byte_arr.getvalue()
    
    def debug_image_info(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Debug function to check image properties for SerpAPI compatibility.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with image information
        """
        try:
            image_path = Path(image_path)
            
            # File info
            file_size = image_path.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            
            # Image info
            with Image.open(image_path) as img:
                width, height = img.size
                mode = img.mode
                format_name = img.format
            
            info = {
                "file_path": str(image_path),
                "file_exists": image_path.exists(),
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size_mb, 2),
                "dimensions": (width, height),
                "mode": mode,
                "format": format_name,
                "suitable_for_serpapi": {
                    "size_ok": file_size_mb <= self.max_file_size_mb,
                    "dimensions_ok": width <= self.max_image_size[0] and height <= self.max_image_size[1],
                    "format_ok": format_name in ['JPEG', 'PNG', 'WebP']
                }
            }
            
            # Check if image needs processing
            needs_processing = (
                file_size_mb > self.max_file_size_mb or 
                width > self.max_image_size[0] or 
                height > self.max_image_size[1] or
                mode in ('RGBA', 'P', 'LA')
            )
            
            info["needs_processing"] = needs_processing
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting image info: {e}")
            return {"error": str(e)}
    
    def detect_objects(self, image_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Detect objects in an image with focus on detection confidence for SerpAPI pipeline.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Dictionary containing detection results and metadata, or None if failed
        """
        start_time = time.time()
        image_path = Path(image_path)
        
        logger.info(f"Starting object detection for: {image_path.name}")
        
        # Validate model is loaded
        if not self.model_loaded or self.model is None:
            logger.error("Model not loaded. Cannot perform detection.")
            return None
        
        # Validate input image
        if not validate_input_image(image_path):
            logger.error(f"Invalid input image: {image_path}")
            return None
        
        try:
            # Run inference with detection-focused parameters
            logger.debug("Running YOLOv8 inference (detection-focused mode)...")
            results = self.model(
                str(image_path),
                conf=self.detection_confidence,  # Lower threshold for detection
                iou=NMS_THRESHOLD,
                verbose=False  # Reduce YOLO's output verbosity
            )
            
            # Process results with quality filtering
            detections = self._process_detection_results_for_serpapi(results[0], image_path)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create enhanced metadata
            if detections:
                metadata = create_enhanced_detection_metadata(
                    image_path, 
                    detections, 
                    YOLO_MODEL_NAME, 
                    processing_time
                )
                logger.info(f"Detection completed: {len(detections)} objects found in {processing_time:.2f}s")
            else:
                metadata = create_empty_enhanced_metadata(
                    image_path, 
                    YOLO_MODEL_NAME, 
                    processing_time
                )
                logger.info(f"Detection completed: No objects found in {processing_time:.2f}s")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error during object detection for {image_path}: {e}")
            return None
    
    def detect_and_prepare_for_serpapi(self, image_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Detect objects and prepare cropped regions for SerpAPI analysis.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Dictionary with detections and SerpAPI-ready image data
        """
        # First run object detection
        detection_results = self.detect_objects(image_path)
        
        if not detection_results or not detection_results.get("detections"):
            logger.info("No objects detected, preparing full image for SerpAPI")
            # Prepare the full image for SerpAPI
            prepared_image = self.prepare_image_for_serpapi(image_path)
            
            return {
                "detection_results": detection_results,
                "serpapi_images": [{
                    "type": "full_image",
                    "base64_data": prepared_image,
                    "description": "Full image (no objects detected)"
                }] if prepared_image else []
            }
        
        # Prepare cropped regions for each detection
        serpapi_images = []
        
        # Always include the full image as backup
        full_image_b64 = self.prepare_image_for_serpapi(image_path)
        if full_image_b64:
            serpapi_images.append({
                "type": "full_image",
                "base64_data": full_image_b64,
                "description": "Full image"
            })
        
        # Add cropped regions for each detection
        original_image = load_image(image_path)
        if original_image is not None:
            for i, detection in enumerate(detection_results["detections"]):
                crop_b64 = self._prepare_crop_for_serpapi(original_image, detection, i)
                if crop_b64:
                    serpapi_images.append(crop_b64)
        
        return {
            "detection_results": detection_results,
            "serpapi_images": serpapi_images
        }
    
    def _prepare_crop_for_serpapi(self, image: np.ndarray, detection: Dict[str, Any], 
                                 crop_id: int) -> Optional[Dict[str, Any]]:
        """
        Prepare a cropped detection region for SerpAPI.
        
        Args:
            image: Original image as numpy array
            detection: Detection dictionary with bounding box
            crop_id: ID for this crop
            
        Returns:
            Dictionary with crop data for SerpAPI, or None if failed
        """
        try:
            bbox = detection["bounding_box"]
            
            # Extract crop
            crop = image[bbox["y1"]:bbox["y2"], bbox["x1"]:bbox["x2"]]
            
            if crop.size == 0:
                logger.warning(f"Empty crop for detection {crop_id}")
                return None
            
            # Convert to PIL Image
            if len(crop.shape) == 3:
                crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            else:
                crop_pil = Image.fromarray(crop)
            
            # Resize if too small (SerpAPI works better with larger images)
            min_size = 200
            if crop_pil.size[0] < min_size or crop_pil.size[1] < min_size:
                # Calculate new size maintaining aspect ratio
                ratio = max(min_size / crop_pil.size[0], min_size / crop_pil.size[1])
                new_size = (int(crop_pil.size[0] * ratio), int(crop_pil.size[1] * ratio))
                crop_pil = crop_pil.resize(new_size, Image.Resampling.LANCZOS)
                logger.debug(f"Upscaled crop {crop_id} to {new_size}")
            
            # Convert to bytes and base64
            crop_bytes = self._image_to_bytes(crop_pil, self.jpeg_quality)
            crop_b64 = base64.b64encode(crop_bytes).decode('utf-8')
            
            description = f"Detected object {crop_id+1}"
            if USE_YOLO_CLASSIFICATION and "yolo_class_name" in detection:
                description += f" ({detection['yolo_class_name']})"
            description += f" - confidence: {detection['detection_confidence']:.3f}"
            
            return {
                "type": "detection_crop",
                "detection_id": detection["detection_id"],
                "base64_data": crop_b64,
                "description": description,
                "bounding_box": bbox,
                "detection_confidence": detection["detection_confidence"]
            }
            
        except Exception as e:
            logger.error(f"Error preparing crop {crop_id} for SerpAPI: {e}")
            return None
    
    def _process_detection_results_for_serpapi(self, result, image_path: Path) -> List[Dict[str, Any]]:
        """
        Process raw YOLOv8 results for SerpAPI pipeline with quality filtering.
        
        Args:
            result: YOLOv8 result object
            image_path: Path to the original image
            
        Returns:
            List of detection dictionaries optimized for SerpAPI
        """
        detections = []
        
        # Check if there are any detections
        if result.boxes is None or len(result.boxes) == 0:
            logger.debug("No objects detected")
            return detections
        
        # Get image dimensions for validation
        image = load_image(image_path)
        if image is None:
            logger.error("Could not load image for processing results")
            return detections
        
        image_height, image_width = image.shape[:2]
        
        # Process each detection with quality filtering
        valid_detections = 0
        for i, box in enumerate(result.boxes):
            try:
                detection = self._extract_detection_info_for_serpapi(
                    box, i, image_width, image_height
                )
                
                if detection:
                    # Apply quality filters
                    if self._passes_quality_filters(detection, image_width, image_height):
                        detections.append(detection)
                        valid_detections += 1
                        
                        if USE_YOLO_CLASSIFICATION:
                            logger.debug(f"Valid detection {valid_detections}: {detection.get('yolo_class_name', 'unknown')} "
                                       f"(detection conf: {detection['detection_confidence']:.3f})")
                        else:
                            logger.debug(f"Valid detection {valid_detections}: object "
                                       f"(detection conf: {detection['detection_confidence']:.3f})")
                    else:
                        logger.debug(f"Detection {i+1} filtered out by quality filters")
                
            except Exception as e:
                logger.warning(f"Error processing detection {i}: {e}")
                continue
        
        logger.info(f"Filtered detections: {valid_detections}/{len(result.boxes)} passed quality filters")
        return detections
    
    def _extract_detection_info_for_serpapi(self, box, detection_id: int, 
                                           image_width: int, image_height: int) -> Optional[Dict[str, Any]]:
        """
        Extract detection information optimized for SerpAPI pipeline.
        
        Args:
            box: YOLOv8 box object
            detection_id: Unique ID for this detection
            image_width: Width of the original image
            image_height: Height of the original image
            
        Returns:
            Detection dictionary optimized for SerpAPI, or None if extraction failed
        """
        try:
            # Extract box coordinates (xyxy format)
            coords = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = coords.astype(int)
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, image_width))
            y1 = max(0, min(y1, image_height))
            x2 = max(0, min(x2, image_width))
            y2 = max(0, min(y2, image_height))
            
            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                logger.warning(f"Invalid bounding box coordinates: [{x1}, {y1}, {x2}, {y2}]")
                return None
            
            # Extract detection confidence (not classification confidence)
            detection_confidence = float(box.conf[0].cpu().numpy())
            
            # Create detection dictionary optimized for SerpAPI
            detection = {
                "detection_id": detection_id,
                "detection_confidence": round(detection_confidence, 4),
                "bounding_box": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "area": (x2 - x1) * (y2 - y1)
                }
            }
            
            # Optionally include YOLO classification (but don't rely on it)
            if USE_YOLO_CLASSIFICATION:
                class_id = int(box.cls[0].cpu().numpy())
                if hasattr(self.model, 'names') and class_id in self.model.names:
                    detection["yolo_class_id"] = class_id
                    detection["yolo_class_name"] = self.model.names[class_id]
                    detection["yolo_class_note"] = "For reference only - SerpAPI will provide accurate classification"
            
            # Add metadata fields if configured
            if METADATA_FIELDS.get("detection_timestamp", False):
                from datetime import datetime
                detection["detection_timestamp"] = datetime.now().isoformat()
            
            return detection
            
        except Exception as e:
            logger.error(f"Error extracting detection info: {e}")
            return None
    
    def _passes_quality_filters(self, detection: Dict[str, Any], 
                               image_width: int, image_height: int) -> bool:
        """
        Apply quality filters to avoid wasting SerpAPI calls on poor detections.
        
        Args:
            detection: Detection dictionary
            image_width: Width of the original image
            image_height: Height of the original image
            
        Returns:
            True if detection passes quality filters
        """
        bbox = detection["bounding_box"]
        
        # Check minimum size
        if (bbox["width"] < QUALITY_FILTERS["min_crop_width"] or 
            bbox["height"] < QUALITY_FILTERS["min_crop_height"]):
            logger.debug(f"Detection {detection['detection_id']} too small: {bbox['width']}x{bbox['height']}")
            return False
        
        # Check if crop is too large (likely background)
        image_area = image_width * image_height
        crop_area = bbox["area"]
        crop_percentage = crop_area / image_area
        
        if crop_percentage > QUALITY_FILTERS["max_crop_percentage"]:
            logger.debug(f"Detection {detection['detection_id']} too large: {crop_percentage:.1%} of image")
            return False
        
        # Check if crop is too small (likely noise)
        if crop_percentage < QUALITY_FILTERS["min_crop_percentage"]:
            logger.debug(f"Detection {detection['detection_id']} too small: {crop_percentage:.1%} of image")
            return False
        
        # Check detection confidence
        if detection["detection_confidence"] < QUALITY_FILTERS["min_detection_confidence"]:
            logger.debug(f"Detection {detection['detection_id']} low confidence: {detection['detection_confidence']:.3f}")
            return False
        
        return True
    
    def get_detection_summary(self, detection_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a summary of detection results for SerpAPI pipeline.
        
        Args:
            detection_results: Detection results from detect_objects()
            
        Returns:
            Summary dictionary optimized for SerpAPI pipeline
        """
        if not detection_results or "detections" not in detection_results:
            return {
                "total_detections": 0,
                "detection_confidence_range": None,
                "yolo_classes_detected": [] if USE_YOLO_CLASSIFICATION else "Not using YOLO classification",
                "processing_time": detection_results.get("processing_info", {}).get("detection_time_seconds", 0)
            }
        
        detections = detection_results["detections"]
        
        # Analyze detection confidences
        detection_confidences = [d["detection_confidence"] for d in detections]
        
        summary = {
            "total_detections": len(detections),
            "detection_confidence_range": {
                "min": min(detection_confidences) if detection_confidences else 0,
                "max": max(detection_confidences) if detection_confidences else 0,
                "avg": sum(detection_confidences) / len(detection_confidences) if detection_confidences else 0
            },
            "processing_time": detection_results.get("processing_info", {}).get("detection_time_seconds", 0),
            "ready_for_serpapi": len(detections)  # All passed quality filters
        }
        
        # Add YOLO classification summary if enabled
        if USE_YOLO_CLASSIFICATION:
            yolo_classes = {}
            for detection in detections:
                if "yolo_class_name" in detection:
                    class_name = detection["yolo_class_name"]
                    yolo_classes[class_name] = yolo_classes.get(class_name, 0) + 1
            
            summary["yolo_classes_detected"] = list(yolo_classes.keys())
            summary["yolo_class_counts"] = yolo_classes
            summary["note"] = "YOLO classifications are for reference only - SerpAPI will provide accurate identification"
        else:
            summary["yolo_classes_detected"] = "Not using YOLO classification - relying on SerpAPI for identification"
        
        return summary
    
    def update_detection_confidence(self, new_threshold: float):
        """
        Update the detection confidence threshold for future detections.
        
        Args:
            new_threshold: New detection confidence threshold (0.0-1.0)
        """
        if not 0.0 <= new_threshold <= 1.0:
            raise ValueError("Detection confidence threshold must be between 0.0 and 1.0")
        
        old_threshold = self.detection_confidence
        self.detection_confidence = new_threshold
        
        logger.info(f"Updated detection confidence threshold: {old_threshold} -> {new_threshold}")
    
    def is_ready(self) -> bool:
        """
        Check if the detector is ready to perform detections.
        
        Returns:
            True if detector is ready, False otherwise
        """
        return self.model_loaded and self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model for SerpAPI pipeline.
        
        Returns:
            Dictionary with model information
        """
        num_classes = len(self.model.names) if hasattr(self.model, 'names') else 0
        
        return {
            "model_name": YOLO_MODEL_NAME,
            "model_path": str(self.model_path),
            "model_loaded": self.model_loaded,
            "detection_confidence_threshold": self.detection_confidence,
            "nms_threshold": NMS_THRESHOLD,
            "detection_mode": DETECTION_MODE,
            "use_yolo_classification": USE_YOLO_CLASSIFICATION,
            "num_classes": num_classes,
            "quality_filters": QUALITY_FILTERS,
            "optimized_for": "SerpAPI integration",
            "image_processing": {
                "max_image_size": self.max_image_size,
                "max_file_size_mb": self.max_file_size_mb,
                "jpeg_quality": self.jpeg_quality
            }
        }

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def detect_objects_simple(image_path: Union[str, Path], 
                         detection_confidence: float = QUALITY_FILTERS["min_detection_confidence"]) -> Optional[Dict[str, Any]]:
    """
    Simple function to detect objects in an image optimized for SerpAPI pipeline.
    
    Args:
        image_path: Path to the input image
        detection_confidence: Detection confidence threshold
        
    Returns:
        Detection results or None if failed
    """
    detector = YOLOv8Detector(detection_confidence=detection_confidence)
    if not detector.is_ready():
        logger.error("Failed to initialize detector")
        return None
    
    return detector.detect_objects(image_path)

def prepare_image_for_serpapi_simple(image_path: Union[str, Path]) -> Optional[str]:
    """
    Simple function to prepare an image for SerpAPI.
    
    Args:
        image_path: Path to the input image
        
    Returns:
        Base64 encoded image string, or None if failed
    """
    detector = YOLOv8Detector()
    return detector.prepare_image_for_serpapi(image_path)

def debug_image_simple(image_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Simple function to debug image properties for SerpAPI.
    
    Args:
        image_path: Path to the input image
        
    Returns:
        Dictionary with image information
    """
    detector = YOLOv8Detector()
    return detector.debug_image_info(image_path)

def test_detector():
    """Test the detector functionality for SerpAPI pipeline."""
    logger.info("Testing YOLOv8 detector (SerpAPI optimized)...")
    
    # Initialize detector
    detector = YOLOv8Detector()
    
    # Check if ready
    if detector.is_ready():
        logger.info("✓ Detector initialized successfully")
        
        # Print model info
        model_info = detector.get_model_info()
        logger.info("Model Information:")
        for key, value in model_info.items():
            if key not in ["quality_filters", "image_processing"]:  # Skip complex objects
                logger.info(f"  {key}: {value}")
        
        # Show quality filters
        logger.info("Quality Filters:")
        for key, value in QUALITY_FILTERS.items():
            logger.info(f"  {key}: {value}")
        
        # Show image processing settings
        logger.info("Image Processing Settings:")
        img_settings = model_info.get("image_processing", {})
        for key, value in img_settings.items():
            logger.info(f"  {key}: {value}")
        
    else:
        logger.error("✗ Detector initialization failed")
    
    return detector.is_ready()

def test_image_processing(image_path: Union[str, Path]):
    """
    Test image processing functionality for SerpAPI.
    
    Args:
        image_path: Path to test image
    """
    logger.info(f"Testing image processing for: {image_path}")
    
    detector = YOLOv8Detector()
    
    if not detector.is_ready():
        logger.error("Detector not ready")
        return
    
    # Debug image info
    logger.info("=== IMAGE DEBUG INFO ===")
    info = detector.debug_image_info(image_path)
    for key, value in info.items():
        logger.info(f"  {key}: {value}")
    
    # Test image preparation
    logger.info("=== TESTING IMAGE PREPARATION ===")
    prepared = detector.prepare_image_for_serpapi(image_path)
    
    if prepared:
        logger.info(f"✓ Image successfully prepared for SerpAPI")
        logger.info(f"✓ Base64 length: {len(prepared)} characters")
        logger.info(f"✓ Estimated size: {len(prepared) * 0.75 / 1024:.1f} KB")
    else:
        logger.error("✗ Failed to prepare image for SerpAPI")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Test the detector
    success = test_detector()
    
    if success:
        print("\n✓ YOLOv8 Detector is ready for SerpAPI integration!")
        print(f"✓ Model: {YOLO_MODEL_NAME}")
        print(f"✓ Detection mode: {DETECTION_MODE}")
        print(f"✓ Detection confidence threshold: {QUALITY_FILTERS['min_detection_confidence']}")
        print(f"✓ Using YOLO classification: {USE_YOLO_CLASSIFICATION}")
        print(f"✓ Quality filters enabled: {len(QUALITY_FILTERS)} filters")
        print(f"✓ Image processing ready for SerpAPI")
        
        # Test image processing if test image is provided
        import sys
        if len(sys.argv) > 1:
            test_image_path = sys.argv[1]
            print(f"\nTesting image processing with: {test_image_path}")
            test_image_processing(test_image_path)
    else:
        print("\n✗ YOLOv8 Detector failed to initialize")
        print("Please check your installation and model files")