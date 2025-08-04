"""
Fixed Product Classification Module - Stops on First Success
Key changes:
2. debug_single_method() - tests only one method at a time
3. Added enable_fallback parameter to control fallback behavior
4. More conservative API usage
"""

import time
import requests
import base64
import io
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from PIL import Image
import hashlib
import os

# Import your existing config and utils
try:
    from .config import (
        SERPAPI_ENGINE,
        SERPAPI_FALLBACK_ENGINE,
        SERPAPI_TIMEOUT,
        EXTRACT_PRODUCT_DETAILS,
        CROP_FILENAME_FALLBACK,
        get_serpapi_key
    )
    from .utils import (
        setup_logging,
        create_product_filename,
        save_image,
        load_image
    )
except ImportError:
    from config import (
        SERPAPI_ENGINE,
        SERPAPI_FALLBACK_ENGINE,
        SERPAPI_TIMEOUT,
        EXTRACT_PRODUCT_DETAILS,
        CROP_FILENAME_FALLBACK,
        get_serpapi_key
    )
    from utils import (
        setup_logging,
        create_product_filename,
        save_image,
        load_image
    )

logger = setup_logging("product_classifier")

class ProductClassifier:
    """
    Credit-conscious Product Classification using SerpAPI
    """
    
    def __init__(self, enable_fallback: bool = False):
        """
        Initialize the Product Classifier.
        
        Args:
            enable_fallback: If False, won't try fallback methods (saves credits)
        """
        self.api_key = get_serpapi_key()
        self.enable_fallback = enable_fallback
        self.total_api_calls = 0
        self.successful_classifications = 0
        self.failed_classifications = 0
        self.total_api_time = 0.0
        
        # Image processing settings
        self.max_image_size = (800, 800)  # Smaller to avoid 414 errors
        self.max_file_size_mb = 1.0
        self.jpeg_quality = 70
        
        logger.info("Initializing Credit-Conscious Product Classifier")
        logger.info(f"Primary engine: {SERPAPI_ENGINE}")
        logger.info(f"Fallback enabled: {self.enable_fallback}")
        
        if not self.api_key:
            logger.error("SerpAPI key not found!")
            raise ValueError("SerpAPI key is required")
        else:
            logger.info("✓ SerpAPI key found")
    
    def test_all_engines(self):
        """Test different SerpAPI engines to see what's available on your plan."""
        
        # Test engines to try
        engines_to_test = [
            ('google', 'Basic Google Search'),
            ('google_lens', 'Google Lens'),
            ('google_reverse_image', 'Google Reverse Image'),
            ('google_images', 'Google Images'),
        ]
        
        print("🧪 Testing SerpAPI Engine Availability...")
        print("=" * 50)
        
        for engine, description in engines_to_test:
            print(f"\n🔍 Testing {description} (engine={engine})")
            
            try:
                if engine == 'google_lens':
                    # For Google Lens, we need an image
                    data = {
                        'engine': engine,
                        'url': 'https://i.imgur.com/test.jpg',  # Test URL
                        'api_key': self.api_key,
                    }
                    response = requests.post('https://serpapi.com/search', data=data, timeout=10)
                    
                elif engine == 'google_reverse_image':
                    # For reverse image, also needs image
                    data = {
                        'engine': engine,
                        'image_url': 'https://i.imgur.com/test.jpg',  # Test URL
                        'api_key': self.api_key,
                    }
                    response = requests.get('https://serpapi.com/search.json', params=data, timeout=10)
                    
                else:
                    # For regular searches
                    params = {
                        'engine': engine,
                        'q': 'test',
                        'api_key': self.api_key,
                    }
                    response = requests.get('https://serpapi.com/search.json', params=params, timeout=10)
                
                print(f"   Status: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    if 'error' in result:
                        print(f"   ❌ API Error: {result['error']}")
                        
                        # Check for specific error messages
                        error_msg = result['error'].lower()
                        if 'not available' in error_msg or 'not supported' in error_msg:
                            print(f"   💡 This engine is not available on your plan")
                        elif 'invalid' in error_msg:
                            print(f"   💡 Invalid parameters for this engine")
                        
                    else:
                        print(f"   ✅ Engine works! Response keys: {list(result.keys())[:5]}")
                        
                elif response.status_code == 404:
                    print(f"   ❌ 404 - Engine not found (likely not available on your plan)")
                    
                elif response.status_code == 403:
                    print(f"   ❌ 403 - Forbidden (plan limitation)")
                    
                else:
                    print(f"   ❌ HTTP {response.status_code}: {response.text[:100]}")
                    
            except Exception as e:
                print(f"   ❌ Exception: {e}")
        
        print("\n" + "=" * 50)
        print("💡 Engines that returned ✅ are available on your plan")
        print("💡 404/403 errors usually mean the engine isn't available on your plan")
    
    def test_api_access(self):
        """Test basic API access with a simple Google search."""
        try:
            params = {
                'engine': 'google',
                'q': 'test search',
                'api_key': self.api_key
            }
            
            response = requests.get('https://serpapi.com/search.json', params=params, timeout=10)
            
            print(f"🧪 API Test Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                if 'error' in result:
                    print(f"❌ API Error: {result['error']}")
                    return False
                else:
                    print(f"✅ API Access Working! Plan info:")
                    if 'search_metadata' in result:
                        print(f"   - Search ID: {result['search_metadata'].get('id', 'N/A')}")
                    return True
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                print(f"Response: {response.text[:200]}")
                return False
                    
        except Exception as e:
            print(f"❌ API Test Failed: {e}")
            return False
   
    def test_google_lens_with_known_image(self):
        """Test Google Lens with a known product image URL."""
        try:
            self.total_api_calls += 1
            
            # Test with a clear product image from the web
            test_url = "https://i.imgur.com/HBrB8p0.png"  # Use the same one from your playground
            
            params = {
                'engine': 'google_lens',
                'url': test_url,
                'api_key': self.api_key,
                'hl': 'en',
            }
            
            logger.info(f"🧪 Testing Google Lens with known image URL")
            
            response = requests.get(
                'https://serpapi.com/search.json',
                params=params,
                timeout=SERPAPI_TIMEOUT
            )
            
            logger.info(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                if 'error' in result:
                    print(f"❌ Error: {result['error']}")
                else:
                    print(f"✅ Success! Response keys: {list(result.keys())}")
                    
                    # Show what we got
                    if 'visual_matches' in result:
                        print(f"📸 Visual matches: {len(result['visual_matches'])}")
                    if 'inline_shopping_results' in result:
                        print(f"🛒 Shopping results: {len(result['inline_shopping_results'])}")
                        
                    return result
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
        
        return None

    def prepare_image_for_serpapi(self, image_path: Path) -> Optional[str]:
        """Prepare image targeting ~10KB for optimal Google Lens results."""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB
                if img.mode in ('RGBA', 'P', 'LA'):
                    img = img.convert('RGB')
                
                # Target size that worked before but with better quality
                max_size = (300, 300)  # Between 200 and 600
                if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                    img.thumbnail(max_size, Image.LANCZOS)
                    logger.debug(f"Resized to: {img.size}")
                
                # Start with decent quality
                quality = 50
                img_bytes = self._image_to_bytes(img, quality)
                
                # Target 8-10KB - bigger than 2.7KB but smaller than 12.7KB
                target_kb = 10
                max_bytes = target_kb * 1024
                
                while len(img_bytes) > max_bytes and quality > 20:
                    quality -= 5
                    img_bytes = self._image_to_bytes(img, quality)
                    logger.debug(f"Quality {quality}%: {len(img_bytes) / 1024:.1f} KB")
                
                # If still too big, reduce size slightly
                if len(img_bytes) > max_bytes:
                    smaller_size = (250, 250)
                    img.thumbnail(smaller_size, Image.LANCZOS)
                    img_bytes = self._image_to_bytes(img, 35)
                    logger.debug(f"Reduced to: {img.size}")
                
                base64_string = base64.b64encode(img_bytes).decode('utf-8')
                
                # Check final sizes
                final_kb = len(img_bytes) / 1024
                data_url_chars = len(f'data:image/jpeg;base64,{base64_string}')
                
                logger.info(f"Image prepared: {final_kb:.1f} KB binary, {data_url_chars} chars data URL")
                
                # Target under 15,000 characters for data URL
                if data_url_chars > 15000:
                    logger.warning(f"Data URL might be too large: {data_url_chars} chars")
                
                return base64_string
                    
        except Exception as e:
            logger.error(f"Failed to prepare image: {e}")
            return None
    
    def _image_to_bytes(self, img: Image.Image, quality: int) -> bytes:
        """Convert PIL Image to bytes with specified JPEG quality."""
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=quality, optimize=True)
        return img_byte_arr.getvalue()

    def test_simple_product_image(self):
        """Test with a simple, clear product image URL."""
        try:
            self.total_api_calls += 1
            
            # Test with a simple product image - iPhone (usually works well)
            test_urls = [
                "https://store.storeimages.cdn-apple.com/4982/as-images.apple.com/is/iphone-15-finish-select-202309-6-1inch-pink?wid=5120&hei=2880&fmt=p-jpg&qlt=80&.v=1692923780378",
                "https://m.media-amazon.com/images/I/61cwywLZR-L.jpg",  # Simple product
                "https://images.unsplash.com/photo-1592921870789-04563d55041c?w=400"  # Simple phone
            ]
            
            for i, test_url in enumerate(test_urls, 1):
                try:
                    params = {
                        'engine': 'google_lens',
                        'url': test_url,
                        'api_key': self.api_key,
                        'hl': 'en',
                    }
                    
                    print(f"🧪 Testing with simple image #{i}")
                    
                    response = requests.get(
                        'https://serpapi.com/search.json',
                        params=params,
                        timeout=30  # Shorter timeout for web images
                    )
                    
                    print(f"Status: {response.status_code}")
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        if 'error' in result:
                            print(f"❌ Error: {result['error']}")
                        else:
                            print(f"✅ SUCCESS! Response keys: {list(result.keys())}")
                            
                            if 'visual_matches' in result:
                                print(f"📸 Visual matches: {len(result['visual_matches'])}")
                            if 'inline_shopping_results' in result:
                                print(f"🛒 Shopping results: {len(result['inline_shopping_results'])}")
                            
                            return result  # Success!
                    else:
                        print(f"❌ HTTP {response.status_code}")
                        
                except Exception as e:
                    print(f"❌ Test {i} failed: {e}")
                    continue
            
            print("❌ All simple image tests failed")
            return None
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            return None


    def extract_product_info(self, serpapi_response: Dict[str, Any], engine_used: str) -> Dict[str, Any]:
        """Extract product info from API response."""
        product_info = {
            "product_name": "Unknown Product",
            "brand": None,
            "confidence": 0.0,
            "serpapi_engine": engine_used,
            "price": None
        }
        
        try:
            if engine_used == 'google_lens':
                # Check visual matches first
                if 'visual_matches' in serpapi_response and serpapi_response['visual_matches']:
                    best_match = serpapi_response['visual_matches'][0]
                    product_info['product_name'] = best_match.get('title', 'Unknown Product')
                    product_info['confidence'] = 0.3  # Set reasonable confidence
                    logger.info(f"Found visual match: {product_info['product_name']}")
                
                # Check shopping results
                elif 'inline_shopping_results' in serpapi_response and serpapi_response['inline_shopping_results']:
                    first_product = serpapi_response['inline_shopping_results'][0]
                    product_info['product_name'] = first_product.get('title', 'Unknown Product')
                    product_info['price'] = first_product.get('price', None)
                    product_info['confidence'] = 0.9
                    logger.info(f"Found shopping result: {product_info['product_name']}")
                
                # If we got any results, consider it successful
                elif 'visual_matches' in serpapi_response or 'related_content' in serpapi_response:
                    logger.info("Google Lens returned results but no clear product matches")
                    # Still try to extract something useful
                    if 'search_metadata' in serpapi_response:
                        product_info['confidence'] = 0.3  # Low but not zero
            
            # Extract brand if we found a product
            if product_info['product_name'] != 'Unknown Product':
                product_info['brand'] = self.extract_brand_from_name(product_info['product_name'])
        
        except Exception as e:
            logger.error(f"Error extracting product info: {e}")
        
        return product_info

    def extract_brand_from_name(self, product_name: str) -> Optional[str]:
        """
        Extract brand from product name with improved logic.
        """
        # Expanded brand list including pharmaceutical/consumer brands
        common_brands = [
            # Tech brands
            'Apple', 'Samsung', 'Google', 'Amazon', 'Microsoft', 'Sony', 'LG', 'Dell',
            'HP', 'Lenovo', 'ASUS', 'Nike', 'Adidas', 'Canon', 'Nikon',
            
            # Pharmaceutical/Health brands
            'TUMS', 'Tums', 'Advil', 'Tylenol', 'Pepto-Bismol', 'Rolaids', 'Alka-Seltzer',
            'Benadryl', 'Claritin', 'Zyrtec', 'Ibuprofen', 'Aspirin',
            
            # Consumer goods brands
            'Coca-Cola', 'Pepsi', 'Nestle', 'Kraft', 'General Mills', 'Kelloggs',
            'Procter & Gamble', 'P&G', 'Johnson & Johnson', 'Unilever'
        ]
        
        product_upper = product_name.upper()
        
        # First, check for exact brand matches (case-insensitive)
        for brand in common_brands:
            if brand.upper() in product_upper:
                return brand
        
        # Look for brand patterns in different parts of the name
        words = product_name.split()
        
        # Check if TUMS/Tums appears anywhere in the name
        for word in words:
            if word.upper() == 'TUMS':
                return 'TUMS'
        
        # Look for brand indicators (words that often come before brand names)
        brand_indicators = ['by', 'from', '|', '-', 'brand:', 'made by']
        for i, word in enumerate(words):
            if word.lower() in brand_indicators and i + 1 < len(words):
                potential_brand = words[i + 1]
                if len(potential_brand) > 2:
                    return potential_brand
        
        # Look for capitalized words that might be brands
        for word in words:
            # If word is all caps and longer than 2 chars, likely a brand
            if word.isupper() and len(word) > 2:
                return word
            # If word starts with capital and has mixed case, might be brand
            elif word[0].isupper() and len(word) > 3 and not word.lower() in ['berry', 'fusion', 'bottle', 'smoothies', 'pack', 'count', 'size']:
                return word
        
        # Last resort: check if first word looks like a brand (not a descriptor)
        first_word = words[0] if words else None
        if first_word and len(first_word) > 2:
            # Skip common product descriptors
            descriptors = ['berry', 'cherry', 'mint', 'original', 'extra', 'super', 'ultra', 'new', 'improved']
            if first_word.lower() not in descriptors:
                return first_word
        
        return None

    def classify_cropped_image(self, cropped_image_path: Union[str, Path]) -> Optional[dict[str, any]]:
        """Main classification method with early exit."""
        return self.call_google_lens_with_cloudinary(cropped_image_path)

    def get_stats(self) -> dict[str, any]:
        """Get classification statistics."""
        return {
            "total_api_calls": self.total_api_calls,
            "successful_classifications": self.successful_classifications,
            "failed_classifications": self.failed_classifications,
            "success_rate": round(
                (self.successful_classifications / max(self.total_api_calls, 1)) * 100, 1
            ),
            "fallback_enabled": self.enable_fallback
        }
    
    #upload to cloudinary
    def upload_to_cloudinary(self, image_path: Path) -> Optional[str]:
        """
        Upload image to Cloudinary and get public URL.
        """
        try:
            # Get Cloudinary credentials
            cloud_name = os.getenv('CLOUDINARY_CLOUD_NAME')
            api_key = os.getenv('CLOUDINARY_API_KEY')
            api_secret = os.getenv('CLOUDINARY_API_SECRET')
            
            if not all([cloud_name, api_key, api_secret]):
                logger.error("Cloudinary credentials not found in .env file")
                logger.error("Please add CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET")
                return None
            
            # Read image file
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()
            
            # Create unique filename
            timestamp = str(int(time.time()))
            filename = f"product_crop_{timestamp}"
            
            # Upload URL
            upload_url = f"https://api.cloudinary.com/v1_1/{cloud_name}/image/upload"
            
            # Create signature (Cloudinary requirement)
            params_to_sign = f"public_id={filename}&timestamp={timestamp}"
            signature = hashlib.sha1(f"{params_to_sign}{api_secret}".encode()).hexdigest()
            
            # Prepare upload
            files = {
                'file': (image_path.name, image_data, 'image/jpeg')
            }
            
            data = {
                'api_key': api_key,
                'timestamp': timestamp,
                'public_id': filename,
                'signature': signature
            }
            
            logger.info(f"📤 Uploading {image_path.name} to Cloudinary...")
            
            response = requests.post(upload_url, files=files, data=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                image_url = result['secure_url']
                logger.info(f"✅ Cloudinary upload successful!")
                logger.info(f"🔗 Public URL: {image_url}")
                return image_url
            else:
                logger.error(f"❌ Cloudinary upload failed: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Cloudinary upload exception: {e}")
            return None

    def classify_full_image_with_cloudinary(self, image_path: Path) -> Optional[dict[str, any]]:
        """
        Upload full image to Cloudinary and classify with Google Lens.
        Used as fallback when YOLO detection fails or crop classification fails.
        """
        try:
            logger.info(f"🖼️ Starting full image classification: {image_path.name}")
            
            # Step 1: Upload full image to Cloudinary
            image_url = self.upload_to_cloudinary(image_path)
            if not image_url:
                logger.error("Failed to upload full image to Cloudinary")
                return None
            
            # Step 2: Call Google Lens with full image URL
            self.total_api_calls += 1
            
            params = {
                'engine': 'google_lens',
                'url': image_url,
                'api_key': self.api_key,
                'hl': 'en',
            }
            
            logger.info(f"🔍 Calling Google Lens with full image (API call #{self.total_api_calls})")
            
            response = requests.get(
                'https://serpapi.com/search.json',
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'error' not in result:
                    product_info = self.extract_product_info(result, 'google_lens')
                    
                    # Mark as full image classification
                    product_info['cloudinary_url'] = image_url
                    product_info['source_image_path'] = str(image_path)
                    product_info['is_full_image'] = True  # ← Key flag
                    product_info['classification_type'] = 'full_image'
                    
                    product_info['classification_metadata'] = {
                        "cloudinary_url": image_url,
                        "api_call_time": round(time.time(), 3),
                        "original_image_path": str(image_path),
                        "classification_timestamp": time.time(),
                        "is_full_image": True
                    }
                    
                    if product_info['product_name'] != 'Unknown Product':
                        logger.info(f"✅ Full image classification SUCCESS: {product_info['product_name']}")
                        self.successful_classifications += 1
                        return product_info
                    else:
                        logger.info("🔍 Full image classification found no clear products")
                        self.failed_classifications += 1
                        return None
                else:
                    logger.error(f"❌ Google Lens error on full image: {result['error']}")
            else:
                logger.error(f"❌ Google Lens failed on full image: {response.status_code}")
            
            self.failed_classifications += 1
            return None
            
        except Exception as e:
            logger.error(f"❌ Full image classification failed: {e}")
            self.failed_classifications += 1
            return None
    
    def call_google_lens_with_cloudinary(self, image_path: Path) -> Optional[Dict[str, Any]]:
        """
        Upload to Cloudinary first, then call Google Lens.
        """
        try:
            # Step 1: Upload to Cloudinary
            image_url = self.upload_to_cloudinary(image_path)
            if not image_url:
                logger.error("Failed to upload image to Cloudinary")
                return None
            
            # Step 2: Call Google Lens with public URL
            self.total_api_calls += 1
            
            params = {
                'engine': 'google_lens',
                'url': image_url,
                'api_key': self.api_key,
                'hl': 'en',
            }
            
            logger.info(f"🔍 Calling Google Lens with Cloudinary URL (API call #{self.total_api_calls})")
            
            response = requests.get(
                'https://serpapi.com/search.json',
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'error' not in result:
                    product_info = self.extract_product_info(result, 'google_lens')
                    
                    # ✅ CRITICAL: Store the Cloudinary URL and metadata
                    product_info['cloudinary_url'] = image_url
                    product_info['source_image_path'] = str(image_path)
                    
                    # ✅ Add timing and success metadata
                    product_info['classification_metadata'] = {
                        "cloudinary_url": image_url,
                        "api_call_time": round(time.time(), 3),
                        "cropped_image_path": str(image_path),
                        "classification_timestamp": time.time()
                    }
                    
                    logger.info(f"✅ Google Lens SUCCESS!")
                    logger.debug(f"Stored Cloudinary URL: {image_url}")
                    return product_info
                else:
                    logger.error(f"❌ Google Lens error: {result['error']}")
            else:
                logger.error(f"❌ Google Lens failed: {response.status_code}")
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Cloudinary + Google Lens failed: {e}")
            return None
        
    def get_classification_stats(self) -> Dict[str, Any]:
        """Get classification statistics."""
        return {
            "total_api_calls": self.total_api_calls,
            "successful_classifications": self.successful_classifications,
            "failed_classifications": self.failed_classifications,
            "success_rate": round(
                (self.successful_classifications / max(self.total_api_calls, 1)) * 100, 1
            ),
            "total_api_time": round(self.total_api_time, 3),
            "average_time_per_call": round(
                self.total_api_time / max(self.total_api_calls, 1), 3
            ),
            "fallback_enabled": self.enable_fallback
        }

# Convenience functions
def classify_single_crop_conservative(cropped_image_path: Union[str, Path], 
                                    enable_fallback: bool = False) -> Optional[dict[str, any]]:
    """
    Conservative classification - stops on first success.
    
    Args:
        cropped_image_path: Path to cropped image
        enable_fallback: If False, won't try fallback methods
    """
    try:
        classifier = ProductClassifier(enable_fallback=enable_fallback)
        return classifier.classify_cropped_image(cropped_image_path)
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        return None

def debug_single_image(image_path: Union[str, Path], method: str = 'google_lens'):
    """
    Debug a single method for testing - saves credits.
    
    Args:
        image_path: Path to test image
        method: 'google_lens' or 'reverse_image'
    """
    try:
        classifier = ProductClassifier(enable_fallback=False)
        classifier.debug_single_method(Path(image_path), method)
    except Exception as e:
        logger.error(f"Debug failed: {e}")



# Main execution
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python product_classifier.py <image_path> [method]")
        print("Methods: google_lens (default), reverse_image")
        sys.exit(1)

    elif sys.argv[1] == "test_api":
        classifier = ProductClassifier(enable_fallback=False)
        classifier.test_api_access()

    elif sys.argv[1] == "test_engines":
        classifier = ProductClassifier(enable_fallback=False)
        classifier.test_all_engines()
    
    elif sys.argv[1] == "test_known":
        classifier = ProductClassifier(enable_fallback=False)
        classifier.test_google_lens_with_known_image()

    elif sys.argv[1] == "test_simple":
        classifier = ProductClassifier(enable_fallback=False)
        classifier.test_simple_product_image()
    
    image_path = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else 'google_lens'
    
    print(f"🧪 Testing SINGLE method to save credits...")
    debug_single_image(image_path, method)