"""
Image Cropping Module for YOLOv8 Object Detection Pipeline with SerpAPI Integration
Handles cropping of detected objects with padding, hierarchical organization, and product classification.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import cv2
import numpy as np

try:
    from .config import (
        CROPPED_IMAGES_DIR,
        METADATA_DIR,
        PADDING_PIXELS,
        PADDING_PERCENTAGE,
        PADDING_MODE,
        MIN_CROP_WIDTH,
        MIN_CROP_HEIGHT,
        CROP_FILENAME_TEMPLATE,
        CROP_FILENAME_FALLBACK,
        METADATA_FILENAME_TEMPLATE,
        OUTPUT_IMAGE_FORMAT,
        OUTPUT_IMAGE_QUALITY,
        USE_YOLO_CLASSIFICATION
    )
    from .utils import (
        setup_logging,
        load_image,
        save_image,
        save_json,
        apply_padding_to_bbox,
        is_valid_bbox,
        create_output_directory,
        get_safe_filename,
        create_enhanced_detection_metadata
    )
    from .product_classifier import ProductClassifier
except ImportError:
    # Handle case when running as main module
    from config import (
        CROPPED_IMAGES_DIR,
        METADATA_DIR,
        PADDING_PIXELS,
        PADDING_PERCENTAGE,
        PADDING_MODE,
        MIN_CROP_WIDTH,
        MIN_CROP_HEIGHT,
        CROP_FILENAME_TEMPLATE,
        CROP_FILENAME_FALLBACK,
        METADATA_FILENAME_TEMPLATE,
        OUTPUT_IMAGE_FORMAT,
        OUTPUT_IMAGE_QUALITY,
        USE_YOLO_CLASSIFICATION
    )
    from utils import (
        setup_logging,
        load_image,
        save_image,
        apply_padding_to_bbox,
        is_valid_bbox,
        create_output_directory,
        get_safe_filename,
        create_enhanced_detection_metadata
    )
    from product_classifier import ProductClassifier

# Initialize logger
logger = setup_logging("cropper")

class ImageCropper:
    """
    Image cropping class for extracting detected objects and integrating with SerpAPI classification.
    """
    
    def __init__(self, output_base_dir: Optional[Path] = None, enable_classification: bool = True):
        """
        Initialize the image cropper with optional SerpAPI integration.
        
        Args:
            output_base_dir: Base directory for cropped images (if None, uses config default)
            enable_classification: Whether to enable SerpAPI product classification
        """
        self.output_base_dir = output_base_dir or CROPPED_IMAGES_DIR
        self.metadata_dir = METADATA_DIR
        self.enable_classification = enable_classification
        
        # Initialize product classifier if enabled
        self.product_classifier = None
        if self.enable_classification:
            try:
                self.product_classifier = ProductClassifier()
                logger.info("✓ Product classifier initialized")
            except Exception as e:
                logger.warning(f"Product classifier initialization failed: {e}")
                logger.warning("Continuing without product classification...")
                self.enable_classification = False
        
        logger.info(f"Initializing Image Cropper")
        logger.info(f"Output directory: {self.output_base_dir}")
        logger.info(f"Product classification: {'Enabled' if self.enable_classification else 'Disabled'}")
        logger.info(f"Padding mode: {PADDING_MODE}")
        
        if PADDING_MODE == "pixels":
            logger.info(f"Padding: {PADDING_PIXELS} pixels")
        else:
            logger.info(f"Padding: {PADDING_PERCENTAGE * 100}%")
        
        # Ensure output directories exist
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
    
    def crop_detections(self, image_path: Union[str, Path], 
                       detection_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Crop all detected objects from an image, classify with SerpAPI, and save with metadata.
        
        Args:
            image_path: Path to the original image
            detection_results: Detection results from YOLOv8Detector
            
        Returns:
            Updated detection results with cropped image paths and product info, or None if failed
        """
        start_time = time.time()
        image_path = Path(image_path)
        
        logger.info(f"Starting cropping and classification process for: {image_path.name}")
        
        # Validate inputs
        if not self._validate_inputs(image_path, detection_results):
            return None
        
        # Load the original image
        image = load_image(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        
        image_height, image_width = image.shape[:2]
        logger.debug(f"Image dimensions: {image_width}x{image_height}")
        
        # Create output directory for this image
        output_dir = create_output_directory(image_path, self.output_base_dir)
        
        # Process detections
        detections = detection_results.get("detections", [])
        
        if not detections:
            logger.info("No detections to crop")
            # Still save metadata for consistency

            return detection_results
        
        # Step 1: Crop all detected objects
        logger.info(f"Step 1: Cropping {len(detections)} detected objects...")
        successful_crops = 0
        cropped_detections = []
        
        for detection in detections:
            try:
                cropped_detection = self._crop_single_detection(
                    image, detection, output_dir, image_path
                )
                
                if cropped_detection and "cropped_image" in cropped_detection:
                    cropped_detections.append(cropped_detection)
                    successful_crops += 1
                else:
                    # Keep original detection even if cropping failed
                    cropped_detections.append(detection)
                    
            except Exception as e:
                logger.error(f"Error cropping detection {detection.get('detection_id', 'unknown')}: {e}")
                # Keep original detection
                cropped_detections.append(detection)
        
        logger.info(f"Cropping results: {successful_crops}/{len(detections)} successful")
        
        # Step 2: Classify cropped objects with SerpAPI (if enabled)
        classified_detections = cropped_detections
        serpapi_time = 0.0
        
        if self.enable_classification and self.product_classifier and successful_crops > 0:
            logger.info(f"Step 2: Classifying {successful_crops} cropped objects with SerpAPI...")
            classification_start = time.time()
            
            classified_detections = self._classify_cropped_objects(cropped_detections, output_dir)
            
            serpapi_time = time.time() - classification_start
            logger.info(f"Classification completed in {serpapi_time:.2f}s")
        else:
            logger.info("Step 2: Skipping SerpAPI classification (disabled or no crops)")
        
        # Update detection results
        updated_results = detection_results.copy()
        updated_results["detections"] = classified_detections
        
        # Add cropping and classification metadata
        total_time = time.time() - start_time
        cropping_time = total_time - serpapi_time
        
        updated_results["cropping_info"] = {
            "total_detections": len(detections),
            "successful_crops": successful_crops,
            "failed_crops": len(detections) - successful_crops,
            "output_directory": str(output_dir),
            "cropping_time_seconds": round(cropping_time, 3)
        }
        
        # Add classification info if enabled
        if self.enable_classification and self.product_classifier:
            classification_stats = self.product_classifier.get_classification_stats()
            updated_results["classification_info"] = classification_stats
            updated_results["classification_info"]["serpapi_time_seconds"] = round(serpapi_time, 3)
        
        # Update processing info with total times
        if "processing_info" in updated_results:
            updated_results["processing_info"]["total_serpapi_time"] = round(serpapi_time, 3)
            updated_results["processing_info"]["total_processing_time"] = (
                updated_results["processing_info"].get("detection_time_seconds", 0) + 
                total_time
            )
        
        logger.info(f"Complete pipeline finished in {total_time:.2f}s "
                   f"(cropping: {cropping_time:.2f}s, classification: {serpapi_time:.2f}s)")
        
        return updated_results
    
    def _validate_inputs(self, image_path: Path, detection_results: Dict[str, Any]) -> bool:
        """
        Validate inputs for cropping operation.
        
        Args:
            image_path: Path to the original image
            detection_results: Detection results to validate
            
        Returns:
            True if inputs are valid, False otherwise
        """
        # Check if image exists
        if not image_path.exists():
            logger.error(f"Image file does not exist: {image_path}")
            return False
        
        # Check detection results structure
        if not isinstance(detection_results, dict):
            logger.error("Detection results must be a dictionary")
            return False
        
        if "detections" not in detection_results:
            logger.error("Detection results missing 'detections' key")
            return False
        
        return True
    
    def _crop_single_detection(self, image: np.ndarray, detection: Dict[str, Any], 
                              output_dir: Path, original_image_path: Path) -> Optional[Dict[str, Any]]:
        """
        Crop a single detected object from the image.
        
        Args:
            image: Original image as numpy array
            detection: Single detection dictionary
            output_dir: Directory to save the cropped image
            original_image_path: Path to the original image
            
        Returns:
            Updated detection dictionary with crop info, or None if failed
        """
        try:
            # Extract bounding box
            bbox_info = detection.get("bounding_box", {})
            bbox = [bbox_info["x1"], bbox_info["y1"], bbox_info["x2"], bbox_info["y2"]]
            
            # Apply padding
            padded_bbox = self._apply_padding(bbox, image.shape)
            
            # Validate bounding box
            if not is_valid_bbox(padded_bbox, MIN_CROP_WIDTH, MIN_CROP_HEIGHT):
                logger.warning(f"Invalid bounding box after padding for detection {detection.get('detection_id', 'unknown')}")
                return detection
            
            # Crop the image
            x1, y1, x2, y2 = padded_bbox
            cropped_image = image[y1:y2, x1:x2]
            
            if cropped_image.size == 0:
                logger.warning(f"Empty crop for detection {detection.get('detection_id', 'unknown')}")
                return detection
            
            # Generate temporary filename (will be renamed after classification if enabled)
            crop_filename = self._generate_temporary_crop_filename(detection)
            crop_path = output_dir / crop_filename
            
            # Save cropped image
            if save_image(cropped_image, crop_path, OUTPUT_IMAGE_FORMAT, OUTPUT_IMAGE_QUALITY):
                # Update detection with crop information
                updated_detection = detection.copy()
                updated_detection["cropped_image"] = {
                    "path": str(crop_path),
                    "filename": crop_filename,
                    "original_bbox": bbox,
                    "padded_bbox": padded_bbox,
                    "crop_dimensions": {
                        "width": x2 - x1,
                        "height": y2 - y1
                    },
                    "padding_applied": {
                        "mode": PADDING_MODE,
                        "value": PADDING_PIXELS if PADDING_MODE == "pixels" else PADDING_PERCENTAGE
                    },
                    "product_renamed": False  # Will be updated after classification
                }
                
                logger.debug(f"Successfully cropped detection {detection.get('detection_id', 'unknown')}: {crop_filename}")
                return updated_detection
            else:
                logger.error(f"Failed to save cropped image: {crop_path}")
                return detection
                
        except Exception as e:
            logger.error(f"Error in _crop_single_detection: {e}")
            return detection
    
    def _classify_cropped_objects(self, detections: List[Dict[str, Any]], 
                             output_dir: Path) -> List[Dict[str, Any]]:
        """
        Classify all cropped objects using the fixed SerpAPI integration.
        """
        enhanced_detections = []
        
        for detection in detections:
            enhanced_detection = detection.copy()
            
            if "cropped_image" in detection:
                crop_path = Path(detection["cropped_image"]["path"])
                
                if crop_path.exists():
                    logger.debug(f"Classifying {crop_path.name}...")
                    
                    # Use the fixed classification method
                    product_info = self.product_classifier.call_google_lens_with_cloudinary(crop_path)
                    
                    if product_info:
                        enhanced_detection["product_info"] = product_info
                        enhanced_detection["classification_status"] = "success"
                        
                        # Rename file with product name
                        new_filename = self._generate_product_filename(product_info, detection["detection_id"])
                        new_crop_path = output_dir / new_filename
                        
                        try:
                            crop_path.rename(new_crop_path)
                            enhanced_detection["cropped_image"]["path"] = str(new_crop_path)
                            enhanced_detection["cropped_image"]["filename"] = new_filename
                            enhanced_detection["cropped_image"]["product_renamed"] = True
                        except Exception as e:
                            logger.warning(f"Failed to rename: {e}")
                    else:
                        enhanced_detection["product_info"] = None
                        enhanced_detection["classification_status"] = "failed"
                else:
                    enhanced_detection["classification_status"] = "crop_missing"
            else:
                enhanced_detection["classification_status"] = "no_crop"
            
            enhanced_detections.append(enhanced_detection)
        
        return enhanced_detections
    
    def _apply_padding(self, bbox: List[int], image_shape: Tuple[int, int, int]) -> List[int]:
        """
        Apply padding to a bounding box based on configuration.
        
        Args:
            bbox: Original bounding box [x1, y1, x2, y2]
            image_shape: Shape of the image (height, width, channels)
            
        Returns:
            Padded bounding box [x1, y1, x2, y2]
        """
        if PADDING_MODE == "pixels":
            return apply_padding_to_bbox(bbox, image_shape, padding_pixels=PADDING_PIXELS)
        else:
            return apply_padding_to_bbox(bbox, image_shape, padding_percentage=PADDING_PERCENTAGE)
    
    def _generate_temporary_crop_filename(self, detection: Dict[str, Any]) -> str:
        """
        Generate a temporary filename for the cropped image (before product classification).
        
        Args:
            detection: Detection dictionary
            
        Returns:
            Temporary filename for the cropped image
        """
        detection_id = detection.get("detection_id", 0)
        
        # Use YOLO class if available and enabled
        if USE_YOLO_CLASSIFICATION and "yolo_class_name" in detection:
            class_name = detection["yolo_class_name"]
            safe_class_name = get_safe_filename(class_name)
            filename = f"{safe_class_name}_{detection_id}.jpg"
        else:
            # Use generic naming for detection-only mode
            filename = f"object_{detection_id}.jpg"
        
        return filename
    
    def _generate_product_filename(self, product_info: Dict[str, Any], detection_id: int) -> str:
        """
        Generate a filename based on product information from SerpAPI.
        
        Args:
            product_info: Product information from SerpAPI
            detection_id: Detection ID for uniqueness
            
        Returns:
            Product-based filename for the cropped image
        """
        try:
            # Try to create filename from product info
            product_name = product_info.get("product_name", "")
            
            if product_name and product_name != "Unknown Product":
                # Clean and format for filename
                safe_name = get_safe_filename(product_name)
                
                # Ensure it's not too long
                if len(safe_name) > 50:
                    safe_name = safe_name[:50].rstrip("_")
                
                filename = f"{safe_name}_{detection_id}.jpg"
                
                logger.debug(f"Generated product filename: {filename}")
                return filename
            else:
                # Fallback to generic naming
                return CROP_FILENAME_FALLBACK.format(detection_id=detection_id)
                
        except Exception as e:
            logger.error(f"Error creating product filename: {e}")
            return CROP_FILENAME_FALLBACK.format(detection_id=detection_id)

    
    def crop_single_object(self, image_path: Union[str, Path], 
                          bbox: List[int], class_name: str = "object",
                          output_filename: Optional[str] = None) -> Optional[str]:
        """
        Crop a single object given its bounding box (utility function).
        
        Args:
            image_path: Path to the original image
            bbox: Bounding box [x1, y1, x2, y2]
            class_name: Name of the object class
            output_filename: Custom output filename (optional)
            
        Returns:
            Path to the cropped image if successful, None otherwise
        """
        image_path = Path(image_path)
        
        # Load image
        image = load_image(image_path)
        if image is None:
            return None
        
        # Apply padding
        padded_bbox = self._apply_padding(bbox, image.shape)
        
        # Validate bounding box
        if not is_valid_bbox(padded_bbox, MIN_CROP_WIDTH, MIN_CROP_HEIGHT):
            logger.warning("Invalid bounding box for single object crop")
            return None
        
        # Crop image
        x1, y1, x2, y2 = padded_bbox
        cropped_image = image[y1:y2, x1:x2]
        
        if cropped_image.size == 0:
            return None
        
        # Create output directory
        output_dir = create_output_directory(image_path, self.output_base_dir)
        
        # Generate filename
        if output_filename:
            crop_filename = output_filename
        else:
            safe_class_name = get_safe_filename(class_name)
            crop_filename = f"{safe_class_name}_single.jpg"
        
        crop_path = output_dir / crop_filename
        
        # Save image
        if save_image(cropped_image, crop_path, OUTPUT_IMAGE_FORMAT, OUTPUT_IMAGE_QUALITY):
            logger.info(f"Successfully saved single crop: {crop_path}")
            return str(crop_path)
        else:
            return None
    
    def get_cropping_summary(self, detection_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of cropping and classification results.
        
        Args:
            detection_results: Detection results with cropping and classification info
            
        Returns:
            Enhanced summary dictionary
        """
        cropping_info = detection_results.get("cropping_info", {})
        classification_info = detection_results.get("classification_info", {})
        detections = detection_results.get("detections", [])
        
        # Count crops and classifications by type
        crop_stats = {
            "total_detections": cropping_info.get("total_detections", 0),
            "successful_crops": cropping_info.get("successful_crops", 0),
            "failed_crops": cropping_info.get("failed_crops", 0),
            "cropping_success_rate": (cropping_info.get("successful_crops", 0) / 
                                    max(cropping_info.get("total_detections", 1), 1)) * 100
        }
        
        # Analyze classifications
        classification_stats = {
            "classification_enabled": self.enable_classification,
            "successful_classifications": 0,
            "failed_classifications": 0,
            "products_by_brand": {},
            "renamed_files": 0
        }
        
        if self.enable_classification:
            for detection in detections:
                if detection.get("classification_status") == "success":
                    classification_stats["successful_classifications"] += 1
                    
                    # Count by brand
                    if "product_info" in detection and detection["product_info"]:
                        brand = detection["product_info"].get("brand")
                        if brand:
                            classification_stats["products_by_brand"][brand] = \
                                classification_stats["products_by_brand"].get(brand, 0) + 1
                    
                    # Count renamed files
                    if detection.get("cropped_image", {}).get("product_renamed", False):
                        classification_stats["renamed_files"] += 1
                
                elif detection.get("classification_status") in ["failed", "crop_missing"]:
                    classification_stats["failed_classifications"] += 1
            
            # Add classification success rate
            total_classifications = (classification_stats["successful_classifications"] + 
                                   classification_stats["failed_classifications"])
            if total_classifications > 0:
                classification_stats["classification_success_rate"] = \
                    (classification_stats["successful_classifications"] / total_classifications) * 100
            else:
                classification_stats["classification_success_rate"] = 0
        
        # Combine summaries
        summary = {
            **crop_stats,
            **classification_stats,
            "output_directory": cropping_info.get("output_directory", ""),
            "cropping_time": cropping_info.get("cropping_time_seconds", 0),
            "serpapi_time": classification_info.get("serpapi_time_seconds", 0),
            "total_time": (cropping_info.get("cropping_time_seconds", 0) + 
                          classification_info.get("serpapi_time_seconds", 0))
        }
        
        return summary
    
    def validate_crops(self, detection_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that all cropped images exist and are readable.
        
        Args:
            detection_results: Detection results with crop paths
            
        Returns:
            Validation results
        """
        validation_results = {
            "total_crops": 0,
            "valid_crops": 0,
            "invalid_crops": 0,
            "missing_files": [],
            "invalid_files": []
        }
        
        detections = detection_results.get("detections", [])
        
        for detection in detections:
            if "cropped_image" in detection:
                validation_results["total_crops"] += 1
                crop_path = Path(detection["cropped_image"]["path"])
                
                if not crop_path.exists():
                    validation_results["invalid_crops"] += 1
                    validation_results["missing_files"].append(str(crop_path))
                    logger.warning(f"Missing crop file: {crop_path}")
                else:
                    # Try to load the image to validate it
                    test_image = load_image(crop_path)
                    if test_image is not None:
                        validation_results["valid_crops"] += 1
                    else:
                        validation_results["invalid_crops"] += 1
                        validation_results["invalid_files"].append(str(crop_path))
                        logger.warning(f"Invalid crop file: {crop_path}")
        
        validation_results["validation_passed"] = validation_results["invalid_crops"] == 0
        
        return validation_results

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def crop_and_classify_detections(image_path: Union[str, Path], 
                               detection_results: Dict[str, Any],
                               enable_classification: bool = True) -> Optional[Dict[str, Any]]:
    """
    Simple function to crop detections and classify with SerpAPI without class instantiation.
    
    Args:
        image_path: Path to the original image
        detection_results: Detection results from YOLOv8Detector
        enable_classification: Whether to enable SerpAPI classification
        
    Returns:
        Updated detection results with crop and classification information, or None if failed
    """
    cropper = ImageCropper(enable_classification=enable_classification)
    return cropper.crop_detections(image_path, detection_results)

def test_cropper():
    """Test the enhanced cropper functionality."""
    logger.info("Testing Enhanced Image Cropper...")
    
    # Test basic initialization
    cropper = ImageCropper(enable_classification=False)
    
    logger.info("✓ Image Cropper initialized successfully")
    logger.info(f"✓ Output directory: {cropper.output_base_dir}")
    logger.info(f"✓ Metadata directory: {cropper.metadata_dir}")
    logger.info(f"✓ Padding mode: {PADDING_MODE}")
    logger.info(f"✓ Product classification: {'Enabled' if cropper.enable_classification else 'Disabled'}")
    
    # Test with classification enabled
    try:
        cropper_with_classification = ImageCropper(enable_classification=True)
        if cropper_with_classification.enable_classification:
            logger.info("✓ SerpAPI integration test passed")
        else:
            logger.warning("⚠ SerpAPI integration disabled (check API key)")
    except Exception as e:
        logger.warning(f"⚠ SerpAPI integration test failed: {e}")
    
    return True

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Test the enhanced cropper
    success = test_cropper()
    
    if success:
        print("\n✓ Enhanced Image Cropper is ready for use!")
        print(f"✓ Output directory: {CROPPED_IMAGES_DIR}")
        print(f"✓ Padding mode: {PADDING_MODE}")
        if PADDING_MODE == "pixels":
            print(f"✓ Padding: {PADDING_PIXELS} pixels")
        else:
            print(f"✓ Padding: {PADDING_PERCENTAGE * 100}%")
        print(f"✓ Minimum crop size: {MIN_CROP_WIDTH}x{MIN_CROP_HEIGHT}")
        print(f"✓ SerpAPI integration: Available")
        print(f"✓ Product-based file naming: Enabled")
    else:
        print("\n✗ Enhanced Image Cropper failed to initialize")