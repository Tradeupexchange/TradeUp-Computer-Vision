#!/usr/bin/env python3
"""
Main script for YOLOv8 Object Detection Pipeline
Runs the complete pipeline: detection -> cropping -> metadata generation
"""


#!/usr/bin/env python3
"""
Main script for YOLOv8 Object Detection Pipeline
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

# DEBUG: Print path information
print(f"🔍 Script location: {Path(__file__)}")
print(f"🔍 Parent directory: {Path(__file__).parent}")
print(f"🔍 Project root: {Path(__file__).parent.parent}")
print(f"🔍 Calculated src path: {Path(__file__).parent.parent / 'src'}")

# Add src to path for imports
src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)

# DEBUG: Check if path was added and file exists
print(f"🔍 Added to Python path: {src_path}")
print(f"🔍 sheets_logger.py exists: {Path(src_path, 'sheets_logger.py').exists()}")
print(f"🔍 Current working directory: {Path.cwd()}")

# Now try the imports
try:
    from config import (
        INPUT_DIR,
        OUTPUT_DIR,
        CROPPED_IMAGES_DIR,
        METADATA_DIR,
        QUALITY_FILTERS,
        YOLO_MODEL_NAME,
        DETECTION_MODE,
        USE_YOLO_CLASSIFICATION,
        validate_paths,
        get_config_summary,
        get_serpapi_key
    )
    print("✅ Config import successful")
except Exception as e:
    print(f"❌ Config import failed: {e}")

try:
    from detector import YOLOv8Detector
    print("✅ Detector import successful")
except Exception as e:
    print(f"❌ Detector import failed: {e}")

try:
    from sheets_logger import GoogleSheetsLogger, log_products_to_sheets
    print("✅ Sheets logger import successful")
except Exception as e:
    print(f"❌ Sheets logger import failed: {e}")



from detector import YOLOv8Detector
from cropper import ImageCropper
from product_classifier import ProductClassifier
from utils import (
    setup_logging,
    validate_input_image,
    is_image_file,
    get_file_info,
    print_enhanced_processing_summary,
    save_json
)

# Initialize logger
logger = setup_logging("main_pipeline")

class DetectionPipeline:
    """
    Main pipeline class that orchestrates the complete detection, cropping, and product classification process.
    """
    
    def __init__(self, detection_confidence: float = 0.35, 
                enable_classification: bool = True, 
                enable_sheets_logging: bool = True):
        """
        Initialize pipeline with Google Sheets integration.
        """
        self.detection_confidence = detection_confidence
        self.enable_classification = enable_classification
        self.enable_sheets_logging = enable_sheets_logging
        self.detector = None
        self.cropper = None
        
        # Initialize statistics with Google Sheets tracking
        self.stats = {
            "total_images": 0,
            "successful_images": 0,
            "failed_images": 0,
            "total_detections": 0,
            "total_crops": 0,
            "total_products_identified": 0,
            "total_sheets_logged": 0,  # ← Add this new stat
            "total_serpapi_calls": 0,
            "total_serpapi_time": 0.0,
            "start_time": None,
            "end_time": None
        }
        
        # Initialize Google Sheets logger
        self.sheets_logger = None
        if self.enable_sheets_logging:
            try:
                from sheets_logger import GoogleSheetsLogger
                self.sheets_logger = GoogleSheetsLogger()
                if self.sheets_logger.connected:
                    logger.info("✅ Google Sheets logging enabled")
                else:
                    logger.warning("⚠️ Google Sheets connection failed - disabling logging")
                    self.enable_sheets_logging = False
            except Exception as e:
                logger.warning(f"⚠️ Google Sheets setup failed: {e}")
                self.enable_sheets_logging = False
        
        logger.info("Initializing Enhanced Detection Pipeline with SerpAPI Integration")
        logger.info(f"Detection confidence: {self.detection_confidence}")
        logger.info(f"Product classification: {'Enabled' if self.enable_classification else 'Disabled'}")
        logger.info(f"Google Sheets logging: {'Enabled' if self.enable_sheets_logging else 'Disabled'}")
        logger.info(f"Detection mode: {DETECTION_MODE}")
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize detector and cropper components with SerpAPI integration."""
        try:
            # Initialize detector with detection-focused approach
            logger.info("Initializing YOLOv8 detector (detection-focused mode)...")
            self.detector = YOLOv8Detector(detection_confidence=self.detection_confidence)
            
            if not self.detector.is_ready():
                raise Exception("YOLOv8 detector failed to initialize")
            
            # Initialize cropper with optional classification
            logger.info(f"Initializing image cropper (classification: {'enabled' if self.enable_classification else 'disabled'})...")
            self.cropper = ImageCropper(enable_classification=self.enable_classification)
            
            # Test SerpAPI if enabled
            if self.enable_classification and self.cropper.enable_classification:
                logger.info("✓ SerpAPI integration ready")
            elif self.enable_classification:
                logger.warning("⚠ SerpAPI integration disabled (check API key)")
                self.enable_classification = False
            
            logger.info("✓ Enhanced pipeline components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline components: {e}")
            raise
    
    def process_image(self, image_path: Path, save_metadata: bool = False) -> Optional[dict]:
        """
        Process a single image through the complete enhanced pipeline with Google Sheets logging.
        
        Args:
            image_path: Path to the input image
            save_metadata: Whether to save metadata files (deprecated - kept for compatibility)
            
        Returns:
            Processing results dictionary or None if failed
        """
        logger.info(f"Processing image: {image_path.name}")
        
        # Validate input
        if not validate_input_image(image_path):
            logger.error(f"Invalid input image: {image_path}")
            return None
        
        try:
            # Step 1: Run object detection (detection-focused)
            logger.info("Step 1: Running object detection (detection-focused mode)...")
            detection_results = self.detector.detect_objects(image_path)
            
            if detection_results is None:
                logger.error("Object detection failed")
                return None
            
            num_detections = len(detection_results.get("detections", []))
            logger.info(f"✓ Detection completed: {num_detections} objects found")
            
            fallback_used = False
            fallback_result = None

            if num_detections > 0:
                # Normal pipeline: YOLO found objects
                if self.enable_classification:
                    logger.info("Step 2: Cropping objects and classifying with SerpAPI...")
                else:
                    logger.info("Step 2: Cropping detected objects...")
                    
                final_results = self.cropper.crop_detections(image_path, detection_results)
                
                if final_results is None:
                    logger.error("Image cropping failed")
                    return None
                
                # Get comprehensive summary
                crop_summary = self.cropper.get_cropping_summary(final_results)
                successful_crops = crop_summary.get("successful_crops", 0)
                successful_classifications = crop_summary.get("successful_classifications", 0)
                crop_summary = {}
                
                if self.enable_classification:
                    logger.info(f"✓ Crop processing completed: {successful_crops} crops, {successful_classifications} products identified")
                    
                    # Check if crop classification failed - trigger fallback
                    if successful_classifications == 0 and successful_crops > 0:
                        logger.info("🔄 Crop classification failed - trying full image fallback...")
                        fallback_result = self._try_full_image_fallback(image_path)
                        if fallback_result:
                            fallback_used = True
                            logger.info("✅ Full image fallback succeeded!")
                        else:
                            logger.info("❌ Full image fallback also failed")
                else:
                    logger.info(f"✓ Cropping completed: {successful_crops} successful crops")

            else:
                # Fallback pipeline: YOLO found 0 objects
                logger.info("Step 2: No objects detected - trying full image classification fallback...")
                
                if self.enable_classification:
                    fallback_result = self._try_full_image_fallback(image_path)
                    if fallback_result:
                        fallback_used = True
                        successful_classifications = 1  # Count the fallback success
                        logger.info("✅ Full image fallback succeeded!")
                    else:
                        successful_classifications = 0
                        logger.info("❌ Full image fallback failed")
                
                # Create minimal results structure for 0 detections
                final_results = detection_results.copy()
                final_results["detections"] = []
                successful_crops = 0
                
                crop_summary = {
                    "successful_crops": 0,
                    "failed_crops": 0,
                    "successful_classifications": successful_classifications,
                    "classification_enabled": self.enable_classification
                }

            if fallback_used and fallback_result:
                # Create a fake detection entry for the full image result
                fake_detection = {
                    "detection_id": "full_image",
                    "product_info": fallback_result,
                    "classification_status": "success",
                    "is_full_image": True,
                    "bounding_box": {"x1": 0, "y1": 0, "x2": 100, "y2": 100}  # Dummy bbox
                }
                final_results["detections"] = [fake_detection]
            
            # Step 3: Log results to Google Sheets (instead of saving files)
            sheets_logged = 0
            if self.enable_sheets_logging and successful_classifications > 0:
                logger.info("📊 Logging results to Google Sheets...")
                sheets_success = self.log_results_to_sheets(final_results, image_path)
                
                if sheets_success:
                    # Count how many products were actually logged
                    sheets_logged = sum(1 for detection in final_results.get("detections", []) 
                                    if detection.get("classification_status") == "success")
                    logger.info(f"✅ {sheets_logged} products logged to Google Sheets")
                else:
                    logger.warning("⚠️ Failed to log results to Google Sheets")
            elif self.enable_sheets_logging and successful_classifications == 0:
                logger.info("📊 No products to log to Google Sheets")
            elif not self.enable_sheets_logging:
                logger.debug("📊 Google Sheets logging disabled")
            
            # Update statistics
            self.stats["total_detections"] += num_detections
            self.stats["total_crops"] += successful_crops
            self.stats["total_products_identified"] += successful_classifications
            self.stats["total_sheets_logged"] = self.stats.get("total_sheets_logged", 0) + sheets_logged
            
            # Update SerpAPI statistics if classification enabled
            if self.enable_classification and "classification_info" in final_results:
                classification_info = final_results["classification_info"]
                self.stats["total_serpapi_calls"] += classification_info.get("total_api_calls", 0)
                self.stats["total_serpapi_time"] += classification_info.get("serpapi_time_seconds", 0)
            
            # Step 4: Generate comprehensive summary (no file saving)
            processing_summary = {
                "image_path": str(image_path),
                "detection_summary": self.detector.get_detection_summary(detection_results),
                "cropping_summary": crop_summary,
                "classification_enabled": self.enable_classification,
                "sheets_logging_enabled": self.enable_sheets_logging,
                "products_logged_to_sheets": sheets_logged,
                "status": "success"
            }
            
            logger.info(f"✅ Successfully processed: {image_path.name}")
            
            # Print per-image summary to terminal
            self._print_image_summary(image_path.name, num_detections, successful_classifications, sheets_logged)
            
            return processing_summary
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return {
                "image_path": str(image_path),
                "status": "failed",
                "error": str(e)
            }
    
    def _try_full_image_fallback(self, image_path: Path) -> Optional[dict[str, any]]:
        """Try classifying the full image as fallback."""
        try:
            if not hasattr(self.cropper, 'product_classifier') or not self.cropper.product_classifier:
                logger.error("No product classifier available for fallback")
                return None
            
            # Use the product classifier to classify the full image
            result = self.cropper.product_classifier.classify_full_image_with_cloudinary(image_path)
            return result
            
        except Exception as e:
            logger.error(f"Full image fallback failed: {e}")
            return None
    
    def process_images(self, image_paths: list) -> dict:
        """
        Process multiple images through the pipeline with Google Sheets integration.
        
        Args:
            image_paths: List of paths to input images
            
        Returns:
            Complete processing results
        """
        self.stats["start_time"] = time.time()
        self.stats["total_images"] = len(image_paths)
        self.stats["total_sheets_logged"] = 0
        
        logger.info(f"Starting batch processing of {len(image_paths)} images")
        if self.enable_sheets_logging:
            logger.info("📊 Google Sheets logging: ENABLED")
        else:
            logger.info("📊 Google Sheets logging: DISABLED")
        
        results = {
            "successful_images": [],
            "failed_images": [],
            "processing_stats": {},
            "config_used": get_config_summary()
        }
        
        # Process each image
        for i, image_path in enumerate(image_paths, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing image {i}/{len(image_paths)}: {image_path.name}")
            logger.info(f"{'='*60}")
            
            result = self.process_image(image_path, save_metadata=False)
            
            if result and result.get("status") == "success":
                results["successful_images"].append(result)
                self.stats["successful_images"] += 1
            else:
                results["failed_images"].append(result or {"image_path": str(image_path), "status": "failed"})
                self.stats["failed_images"] += 1
        
        # Calculate final statistics
        self.stats["end_time"] = time.time()
        processing_time = self.stats["end_time"] - self.stats["start_time"]
        
        results["processing_stats"] = {
            "total_images": self.stats["total_images"],
            "successful_images": self.stats["successful_images"],
            "failed_images": self.stats["failed_images"],
            "success_rate": (self.stats["successful_images"] / self.stats["total_images"]) * 100,
            "total_detections": self.stats["total_detections"],
            "total_crops": self.stats["total_crops"],
            "total_products_identified": self.stats["total_products_identified"],
            "total_sheets_logged": self.stats.get("total_sheets_logged", 0),
            "product_identification_rate": (self.stats["total_products_identified"] / max(self.stats["total_crops"], 1)) * 100,
            "sheets_logging_rate": (self.stats.get("total_sheets_logged", 0) / max(self.stats["total_products_identified"], 1)) * 100,
            "total_serpapi_calls": self.stats["total_serpapi_calls"],
            "total_serpapi_time": round(self.stats["total_serpapi_time"], 2),
            "processing_time_seconds": round(processing_time, 2),
            "average_time_per_image": round(processing_time / self.stats["total_images"], 2),
            "classification_enabled": self.enable_classification,
            "sheets_logging_enabled": self.enable_sheets_logging
        }
        
        # Print enhanced summary with Google Sheets info
        self._print_enhanced_processing_summary(processing_time)

        return results
    
    def _print_enhanced_processing_summary(self, processing_time: float):
        """Print enhanced processing summary with Google Sheets integration."""
        print("\n" + "="*70)
        print("🚀 ENHANCED PROCESSING SUMMARY")
        print("="*70)
        print(f"📸 Images processed: {self.stats['total_images']} "
            f"(✅ {self.stats['successful_images']}, ❌ {self.stats['failed_images']})")
        print(f"🔍 Objects detected: {self.stats['total_detections']}")
        print(f"🎯 Products identified: {self.stats['total_products_identified']}/{self.stats['total_crops']}")
        
        if self.stats['total_crops'] > 0:
            identification_rate = (self.stats['total_products_identified'] / self.stats['total_crops']) * 100
            print(f"📈 Identification success rate: {identification_rate:.1f}%")
        
        # Google Sheets specific stats
        if self.enable_sheets_logging:
            sheets_logged = self.stats.get('total_sheets_logged', 0)
            print(f"📊 Products logged to Google Sheets: {sheets_logged}")
            if self.stats['total_products_identified'] > 0:
                sheets_rate = (sheets_logged / self.stats['total_products_identified']) * 100
                print(f"📋 Sheets logging success rate: {sheets_rate:.1f}%")
        else:
            print(f"📊 Google Sheets logging: DISABLED")
        
        print("-" * 70)
        print(f"⏱️  YOLO detection time: {processing_time - self.stats['total_serpapi_time']:.2f}s")
        print(f"🤖 SerpAPI classification time: {self.stats['total_serpapi_time']:.2f}s")
        print(f"⏰ Total processing time: {processing_time:.2f}s")
        print(f"📊 Average time per image: {processing_time / self.stats['total_images']:.2f}s")
        print("-" * 70)
        print(f"🔗 SerpAPI calls made: {self.stats['total_serpapi_calls']}")
        if self.stats['total_serpapi_calls'] > 0:
            avg_serpapi_time = self.stats['total_serpapi_time'] / self.stats['total_serpapi_calls']
            print(f"⚡ Average SerpAPI time per call: {avg_serpapi_time:.2f}s")
        
        # Show Google Sheets summary if enabled
        if self.enable_sheets_logging and hasattr(self, 'sheets_logger') and self.sheets_logger:
            sheet_summary = self.sheets_logger.get_sheet_summary()
            if sheet_summary.get('connected'):
                print("-" * 70)
                print(f"📈 Google Sheets Summary:")
                print(f"   📋 Total products in sheet: {sheet_summary.get('total_products', 0)}")
                brands = sheet_summary.get('brands_detected', {})
                if brands:
                    print(f"   🏷️  Brands detected: {', '.join(f'{brand}({count})' for brand, count in brands.items())}")
        
        print("="*70)
    
    def _print_image_summary(self, image_name: str, detections: int, products: int, sheets_logged: int):
        """Print a summary for each processed image."""
        print(f"\n📋 {image_name}:")
        print(f"   🔍 Detections: {detections}")
        print(f"   🎯 Products identified: {products}")
        if self.enable_sheets_logging:
            print(f"   📊 Logged to sheets: {sheets_logged}")
        print(f"   ✅ Status: Complete")

    def log_results_to_sheets(self, detection_results: dict[str, any], 
                            original_image_path: Path) -> bool:
        """Log detection results to Google Sheets."""
        if not self.enable_sheets_logging or not self.sheets_logger:
            return False
        
        try:
            summary = self.sheets_logger.log_detection_batch(
                detection_results, 
                original_image_path.name
            )
            
            logger.debug(f"📊 Google Sheets logging summary: {summary}")
            return summary['logged'] > 0
            
        except Exception as e:
            logger.error(f"Failed to log to Google Sheets: {e}")
            return False

    
    def run_multi_threshold_analysis(self, image_path: Path) -> dict:
        """
        Run detection with multiple detection thresholds for analysis.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Multi-threshold analysis results
        """
        logger.info(f"Running multi-threshold detection analysis for: {image_path.name}")
        
        results = self.detector.detect_with_multiple_thresholds(image_path)
        
        # Print comparison
        print("\n" + "="*70)
        print("MULTI-THRESHOLD DETECTION ANALYSIS")
        print("="*70)
        print(f"Image: {image_path.name}")
        print(f"Mode: {DETECTION_MODE} (ignoring YOLO classification)")
        print("-"*70)
        
        for threshold_name, threshold_data in results.items():
            threshold_value = threshold_data["threshold"]
            detection_count = threshold_data["detections_count"]
            print(f"{threshold_name:15} (conf={threshold_value:.2f}): {detection_count:2d} detections")
        
        print("-"*70)
        print("Note: All detections would be sent to SerpAPI for classification")
        print("="*70)
        
        return results

    def debug_detections(self, detection_results):
        """Debug what YOLO detected."""
        print("\n" + "="*60)
        print("DETECTION DEBUG INFO")
        print("="*60)
        
        detections = detection_results.get("detections", [])
        print(f"Total detections: {len(detections)}")
        
        for i, detection in enumerate(detections, 1):
            print(f"\nDetection {i}:")
            print(f"  ID: {detection.get('detection_id', 'N/A')}")
            print(f"  Class: {detection.get('yolo_class_name', 'N/A')}")
            print(f"  Confidence: {detection.get('confidence', 'N/A'):.3f}")
            
            bbox = detection.get('bounding_box', {})
            print(f"  Bbox: [{bbox.get('x1', 0)}, {bbox.get('y1', 0)}, {bbox.get('x2', 0)}, {bbox.get('y2', 0)}]")
            
            if 'cropped_image' in detection:
                crop_path = detection['cropped_image'].get('filename', 'N/A')
                print(f"  Crop: {crop_path}")
            
            if 'product_info' in detection and detection['product_info']:
                product_name = detection['product_info'].get('product_name', 'N/A')
                brand = detection['product_info'].get('brand', 'N/A')
                print(f"  Product: {product_name}")
                print(f"  Brand: {brand}")
        
        print("="*60)

def setup_argument_parser():
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="YOLOv8 Object Detection Pipeline with SerpAPI Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single image with SerpAPI classification
  python run_detection.py input/image.jpg
  
  # Process with custom detection confidence
  python run_detection.py input/image.jpg --confidence 0.15
  
  # Process without SerpAPI classification (cropping only)
  python run_detection.py input/image.jpg --no-classification
  
  # Process multiple images
  python run_detection.py input/image1.jpg input/image2.jpg
  
  # Run multi-threshold detection analysis
  python run_detection.py input/image.jpg --multi-threshold
  
  # Process all images in a directory
  python run_detection.py input/ --batch
        """
    )
    
    parser.add_argument(
        "input",
        nargs="+",
        help="Input image path(s) or directory"
    )
    
    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=QUALITY_FILTERS["min_detection_confidence"],
        help=f"Detection confidence threshold (default: {QUALITY_FILTERS['min_detection_confidence']})"
    )
    
    parser.add_argument(
        "--no-classification",
        action="store_true",
        help="Disable SerpAPI product classification (cropping only)"
    )
    
    parser.add_argument(
        "--multi-threshold", "-m",
        action="store_true",
        help="Run multi-threshold detection analysis (only works with single image)"
    )
    
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="Process all images in directory (if input is directory)"
    )
    
    parser.add_argument(
        "--no-reports",
        action="store_true",
        help="Don't save processing reports"
    )
    
    parser.add_argument(
        "--validate-setup",
        action="store_true",
        help="Validate pipeline setup and configuration"
    )
    
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show configuration information and exit"
    )

    parser.add_argument(
        "--no-sheets",
        action="store_true",
        help="Disable Google Sheets logging (products will not be logged to sheet)"
    )
    
    return parser

def collect_image_paths(input_paths: list, batch_mode: bool = False) -> list:
    """
    Collect image paths from input arguments.
    
    Args:
        input_paths: List of input paths from command line
        batch_mode: Whether to process directories in batch mode
        
    Returns:
        List of valid image paths
    """
    image_paths = []
    
    for input_path in input_paths:
        path = Path(input_path)
        
        if path.is_file():
            if is_image_file(path):
                image_paths.append(path)
            else:
                logger.warning(f"Skipping non-image file: {path}")
        
        elif path.is_dir() and batch_mode:
            # Find all image files in directory
            found_images = []
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]:
                found_images.extend(path.glob(f"*{ext}"))
                found_images.extend(path.glob(f"*{ext.upper()}"))
            
            if found_images:
                image_paths.extend(found_images)
                logger.info(f"Found {len(found_images)} images in directory: {path}")
            else:
                logger.warning(f"No images found in directory: {path}")
        
        else:
            logger.error(f"Invalid input path: {path}")
    
    return image_paths

def validate_pipeline_setup():
    """Validate enhanced pipeline setup and configuration."""
    print("Validating Enhanced YOLOv8 Detection Pipeline Setup...")
    print("="*60)
    
    # Validate configuration
    try:
        from config import validate_config
        validate_config()
        print("✓ Configuration validation passed")
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
        return False
    
    # Validate and create paths
    try:
        validate_paths()
        print("✓ Directory structure validated")
    except Exception as e:
        print(f"✗ Directory validation failed: {e}")
        return False
    
    # Test detector initialization
    try:
        detector = YOLOv8Detector()
        if detector.is_ready():
            print("✓ YOLOv8 detector initialized successfully")
            print(f"  - Model: {detector.get_model_info()['model_name']}")
            print(f"  - Detection mode: {DETECTION_MODE}")
            print(f"  - Using YOLO classification: {USE_YOLO_CLASSIFICATION}")
        else:
            print("✗ YOLOv8 detector failed to initialize")
            return False
    except Exception as e:
        print(f"✗ Detector initialization failed: {e}")
        return False
    
    # Test cropper initialization (without classification)
    try:
        cropper = ImageCropper(enable_classification=False)
        print("✓ Image cropper initialized successfully")
    except Exception as e:
        print(f"✗ Cropper initialization failed: {e}")
        return False
    
    # Test SerpAPI integration
    try:
        api_key = get_serpapi_key()
        if api_key:
            cropper_with_serpapi = ImageCropper(enable_classification=True)
            if cropper_with_serpapi.enable_classification:
                print("✓ SerpAPI integration ready")
                print(f"  - API key: Found")
                print(f"  - Product classification: Enabled")
            else:
                print("⚠ SerpAPI integration disabled")
                print(f"  - Check API key configuration")
        else:
            print("⚠ SerpAPI API key not found")
            print(f"  - Product classification will be disabled")
            print(f"  - Set SERPAPI_API_KEY in .env file to enable")
    except Exception as e:
        print(f"⚠ SerpAPI integration test failed: {e}")
    
    print("="*60)
    print("✓ Enhanced pipeline setup validation COMPLETED")
    return True

def show_configuration_info():
    """Show enhanced configuration information."""
    print("Enhanced YOLOv8 Detection Pipeline Configuration")
    print("="*60)
    
    config = get_config_summary()
    for key, value in config.items():
        print(f"{key:25}: {value}")
    
    print(f"\nSerpAPI Configuration:")
    api_key = get_serpapi_key()
    print(f"{'API Key Status':<25}: {'Found' if api_key else 'Not Found'}")
    if api_key:
        print(f"{'Product Classification':<25}: Enabled")
    else:
        print(f"{'Product Classification':<25}: Disabled (no API key)")
    
    print("\nDirectory Structure:")
    print(f"{'Input':<25}: {INPUT_DIR}")
    print(f"{'Output':<25}: {OUTPUT_DIR}")
    print(f"{'Cropped Images':<25}: {CROPPED_IMAGES_DIR}")
    print(f"{'Metadata':<25}: {METADATA_DIR}")
    
    print("\nQuality Filters:")
    for name, value in QUALITY_FILTERS.items():
        print(f"  {name:<22}: {value}")

def main():
    """Main function."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Handle info flag
    if args.info:
        show_configuration_info()
        return 0
    
    # Handle validation flag
    if args.validate_setup:
        success = validate_pipeline_setup()
        return 0 if success else 1
    
    # Validate confidence threshold
    if not 0.0 <= args.confidence <= 1.0:
        print("Error: Confidence threshold must be between 0.0 and 1.0")
        return 1
    
    # Collect image paths
    image_paths = collect_image_paths(args.input, args.batch)
    
    if not image_paths:
        print("Error: No valid image files found")
        return 1
    
    try:
        # Initialize enhanced pipeline with Google Sheets integration
        enable_classification = not args.no_classification
        enable_sheets = not args.no_sheets  # ← Add this line
        
        pipeline = DetectionPipeline(
            detection_confidence=args.confidence,
            enable_classification=enable_classification,
            enable_sheets_logging=enable_sheets  # ← Add this parameter
        )
        
        # Handle multi-threshold analysis
        if args.multi_threshold:
            if len(image_paths) > 1:
                print("Warning: Multi-threshold analysis only works with single image")
                print("Using first image only")
            
            pipeline.run_multi_threshold_analysis(image_paths[0])
            return 0
        
        # Process images (save_reports now defaults to False since we're using Google Sheets)
        results = pipeline.process_images(image_paths)
        
        # Show Google Sheets summary if enabled
        if enable_sheets and hasattr(pipeline, 'sheets_logger') and pipeline.sheets_logger:
            if pipeline.sheets_logger.connected:
                print("\n📊 Google Sheets Summary:")
                sheet_summary = pipeline.sheets_logger.get_sheet_summary()
                if sheet_summary.get('connected'):
                    print(f"   📋 Total products in sheet: {sheet_summary.get('total_products', 0)}")
                    brands = sheet_summary.get('brands_detected', {})
                    if brands:
                        print(f"   🏷️  Brands detected: {', '.join(f'{brand}({count})' for brand, count in brands.items())}")
                    print(f"   🔗 Spreadsheet ID: {pipeline.sheets_logger.spreadsheet_id}")
                    print(f"   📅 Last updated: {sheet_summary.get('last_updated', 'Unknown')}")
        
        # Optional: Remove the debug section since we have proper logging now
        # if results["successful_images"]:
        #     for result in results["successful_images"]:
        #         if "detection_summary" in result:
        #             print("\n🔍 Debugging detection results...")

        # Return appropriate exit code
        processing_stats = results.get("processing_stats", {})
        failed_images = processing_stats.get("failed_images", 0)
        
        if failed_images > 0:
            logger.warning(f"Pipeline completed with {failed_images} failed images")
            return 1
        else:
            logger.info("Pipeline completed successfully")
            return 0

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())