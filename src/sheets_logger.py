"""
Google Sheets Logger Module
Handles logging product detection results to Google Sheets automatically
"""

import os
import gspread
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import time

try:
    from .config import get_config_summary
    from .utils import setup_logging
except ImportError:
    from config import get_config_summary
    from utils import setup_logging

logger = setup_logging("sheets_logger")

class GoogleSheetsLogger:
    """
    Google Sheets integration for product detection pipeline.
    Automatically logs detected products with their Cloudinary URLs.
    """
    
    def __init__(self):
        """Initialize Google Sheets connection."""
        self.gc = None
        self.sheet = None
        self.spreadsheet_id = None
        self.connected = False
        
        # Get credentials from environment
        self.key_file = os.getenv('GOOGLE_SHEETS_KEY_FILE', 'service-account-key.json')
        self.spreadsheet_id = os.getenv('GOOGLE_SHEETS_SPREADSHEET_ID')
        
        # Initialize connection
        self._connect()
    
    def _connect(self):
        """Establish connection to Google Sheets."""
        try:
            # Check if key file exists
            if not Path(self.key_file).exists():
                logger.error(f"Google Sheets key file not found: {self.key_file}")
                logger.error("Please ensure service-account-key.json is in your project root")
                return
            
            # Check if spreadsheet ID is set
            if not self.spreadsheet_id:
                logger.error("GOOGLE_SHEETS_SPREADSHEET_ID not found in .env file")
                logger.error("Please add your Google Sheets ID to .env")
                return
            
            # Connect to Google Sheets
            logger.info("Connecting to Google Sheets...")
            self.gc = gspread.service_account(filename=self.key_file)
            self.sheet = self.gc.open_by_key(self.spreadsheet_id).sheet1
            
            # Test the connection
            self._test_connection()
            
            self.connected = True
            logger.info("✅ Google Sheets connection established")
            
        except gspread.exceptions.SpreadsheetNotFound:
            logger.error(f"❌ Spreadsheet not found: {self.spreadsheet_id}")
            logger.error("Check that the spreadsheet ID is correct and shared with your service account")
        except gspread.exceptions.APIError as e:
            logger.error(f"❌ Google Sheets API error: {e}")
        except Exception as e:
            logger.error(f"❌ Google Sheets connection failed: {e}")
    
    def _test_connection(self):
        """Test the Google Sheets connection."""
        try:
            # Try to read the first cell
            test_value = self.sheet.cell(1, 1).value
            logger.debug(f"Connection test successful. A1 = '{test_value}'")
        except Exception as e:
            logger.warning(f"Connection test failed: {e}")
            raise
    
    def setup_headers(self):
        """Set up column headers if they don't exist."""
        if not self.connected:
            logger.error("Not connected to Google Sheets")
            return False
        
        try:
            # Check if headers already exist
            headers = self.sheet.row_values(1)
            
            expected_headers = [
                "Product Name", 
                "Cloudinary URL", 
                "Brand", 
                "Confidence", 
                "Detection Date",
                "Original Image",
                "YOLO Class"
            ]
            
            # Set headers if row is empty or doesn't match
            if not headers or headers != expected_headers:
                logger.info("Setting up Google Sheets headers...")
                self.sheet.update('A1:G1', [expected_headers])
                
                # Format headers (bold)
                self.sheet.format('A1:G1', {
                    'textFormat': {'bold': True},
                    'backgroundColor': {'red': 0.9, 'green': 0.9, 'blue': 0.9}
                })
                
                logger.info("✅ Headers set up successfully")
            else:
                logger.debug("Headers already exist and are correct")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set up headers: {e}")
            return False
    
    def log_product(self, product_info: Dict[str, Any], cloudinary_url: str, 
                   original_image: str = "", yolo_class: str = "") -> bool:
        """
        Log a single product detection to Google Sheets.
        
        Args:
            product_info: Product information from classification
            cloudinary_url: Cloudinary public URL for the cropped image
            original_image: Name of the original image
            yolo_class: YOLO detected class name
            
        Returns:
            True if logged successfully, False otherwise
        """
        if not self.connected:
            logger.error("Not connected to Google Sheets")
            return False
        
        try:
            # Extract product details
            product_name = product_info.get('product_name', 'Unknown Product')
            brand = product_info.get('brand', '')
            confidence = product_info.get('confidence', 0)
            detection_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Prepare row data
            row_data = [
                product_name,
                cloudinary_url,
                brand,
                f"{confidence:.2f}" if confidence else "",
                detection_date,
                original_image,
                yolo_class
            ]
            
            # Append to sheet
            self.sheet.append_row(row_data)
            
            logger.info(f"✅ Logged to Google Sheets: {product_name}")
            logger.debug(f"   Brand: {brand}")
            logger.debug(f"   Confidence: {confidence:.2f}")
            logger.debug(f"   URL: {cloudinary_url}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to log product to Google Sheets: {e}")
            return False
    
    def log_detection_batch(self, detection_results: Dict[str, Any], 
                           original_image_name: str) -> Dict[str, Any]:
        """
        Log all products from a detection batch to Google Sheets.
        
        Args:
            detection_results: Complete detection results
            original_image_name: Name of the original image
            
        Returns:
            Summary of logging results
        """
        if not self.connected:
            logger.error("Cannot log batch - not connected to Google Sheets")
            return {"logged": 0, "failed": 0, "skipped": 0}
        
        # Ensure headers are set up
        self.setup_headers()
        
        summary = {"logged": 0, "failed": 0, "skipped": 0}
        detections = detection_results.get("detections", [])
        
        logger.info(f"Logging {len(detections)} detections to Google Sheets...")
        
        for detection in detections:
            try:
                # Check if this detection has product info and Cloudinary URL
                product_info = detection.get("product_info")
                classification_status = detection.get("classification_status")
                
                if classification_status != "success" or not product_info:
                    logger.debug(f"Skipping detection {detection.get('detection_id', 'unknown')}: "
                               f"status = {classification_status}")
                    summary["skipped"] += 1
                    continue
                
                # Get Cloudinary URL from the classification process
                cloudinary_url = self._extract_cloudinary_url(detection)
                
                if not cloudinary_url:
                    logger.warning(f"No Cloudinary URL found for detection {detection.get('detection_id', 'unknown')}")
                    summary["failed"] += 1
                    continue
                
                # Get YOLO class name
                yolo_class = detection.get("yolo_class_name", "")
                
                # Log to sheets
                success = self.log_product(
                    product_info=product_info,
                    cloudinary_url=cloudinary_url,
                    original_image=original_image_name,
                    yolo_class=yolo_class
                )
                
                if success:
                    summary["logged"] += 1
                else:
                    summary["failed"] += 1
                    
                # Small delay to avoid rate limits
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error logging detection: {e}")
                summary["failed"] += 1
        
        # Log summary
        logger.info(f"Google Sheets logging complete: "
                   f"{summary['logged']} logged, "
                   f"{summary['failed']} failed, "
                   f"{summary['skipped']} skipped")
        
        return summary
    
    def _extract_cloudinary_url(self, detection: Dict[str, Any]) -> Optional[str]:
        """
        Extract Cloudinary URL from detection results.
        This assumes the ProductClassifier stores the Cloudinary URL somewhere.
        """
        # Check if it's stored in product_info
        product_info = detection.get("product_info", {})
        
        # Look for Cloudinary URL in various possible locations
        possible_keys = ["cloudinary_url", "image_url", "source_url", "upload_url"]
        
        for key in possible_keys:
            if key in product_info and product_info[key]:
                url = product_info[key]
                if "cloudinary.com" in url:
                    return url
        
        # If not found in product_info, check classification metadata
        classification_metadata = product_info.get("classification_metadata", {})
        for key in possible_keys:
            if key in classification_metadata:
                url = classification_metadata[key]
                if "cloudinary.com" in url:
                    return url
        
        # Check if it's stored at the detection level
        for key in possible_keys:
            if key in detection:
                url = detection[key]
                if "cloudinary.com" in url:
                    return url
        
        logger.warning("Could not find Cloudinary URL in detection results")
        return None
    
    def clear_sheet(self) -> bool:
        """Clear all data from the sheet (keep headers)."""
        if not self.connected:
            return False
        
        try:
            # Get all values
            all_values = self.sheet.get_all_values()
            
            if len(all_values) > 1:  # More than just headers
                # Clear everything except headers
                last_row = len(all_values)
                self.sheet.batch_clear([f"A2:G{last_row}"])
                logger.info(f"Cleared {last_row - 1} rows from Google Sheets")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear sheet: {e}")
            return False
    
    def get_sheet_summary(self) -> Dict[str, Any]:
        """Get summary information about the sheet."""
        if not self.connected:
            return {"connected": False}
        
        try:
            all_values = self.sheet.get_all_values()
            row_count = len(all_values) - 1  # Exclude header
            
            # Count by brand if possible
            brands = {}
            if row_count > 0:
                for row in all_values[1:]:  # Skip header
                    if len(row) >= 3:  # Has brand column
                        brand = row[2] or "Unknown"
                        brands[brand] = brands.get(brand, 0) + 1
            
            return {
                "connected": True,
                "spreadsheet_id": self.spreadsheet_id,
                "total_products": row_count,
                "brands_detected": brands,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            logger.error(f"Failed to get sheet summary: {e}")
            return {"connected": False, "error": str(e)}

# Convenience functions
def create_sheets_logger() -> Optional[GoogleSheetsLogger]:
    """Create a Google Sheets logger instance."""
    try:
        return GoogleSheetsLogger()
    except Exception as e:
        logger.error(f"Failed to create Google Sheets logger: {e}")
        return None

def log_products_to_sheets(detection_results: Dict[str, Any], 
                          original_image_name: str) -> bool:
    """
    Simple function to log detection results to Google Sheets.
    
    Args:
        detection_results: Complete detection results
        original_image_name: Name of the original image
        
    Returns:
        True if logging was successful, False otherwise
    """
    sheets_logger = create_sheets_logger()
    if not sheets_logger or not sheets_logger.connected:
        return False
    
    summary = sheets_logger.log_detection_batch(detection_results, original_image_name)
    return summary["logged"] > 0

def test_sheets_connection() -> bool:
    """Test Google Sheets connection."""
    logger.info("Testing Google Sheets connection...")
    
    sheets_logger = create_sheets_logger()
    if not sheets_logger:
        return False
    
    if sheets_logger.connected:
        # Test logging a sample product
        test_product = {
            "product_name": "Test Product - " + datetime.now().strftime("%H:%M:%S"),
            "brand": "Test Brand",
            "confidence": 0.95
        }
        
        success = sheets_logger.log_product(
            product_info=test_product,
            cloudinary_url="https://test.cloudinary.com/test.jpg",
            original_image="test_image.jpg",
            yolo_class="bottle"
        )
        
        if success:
            logger.info("✅ Google Sheets test successful!")
            return True
        else:
            logger.error("❌ Google Sheets test failed")
            return False
    else:
        logger.error("❌ Google Sheets connection failed")
        return False

if __name__ == "__main__":
    # Test the Google Sheets integration
    success = test_sheets_connection()
    
    if success:
        print("\n✅ Google Sheets integration is working!")
        print("Ready to log product detections automatically")
    else:
        print("\n❌ Google Sheets integration failed")
        print("Check your .env file and service account setup")