from PIL import Image
import io
import aiohttp
from typing import Optional

async def verify_image(image_url: str) -> bool:
    """Verify if URL points to a valid image with basic quality requirements"""
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.get(image_url) as response:
                    if response.status != 200:
                        print(f"Image verification failed: Status {response.status} for {image_url}")
                        return False

                    # Check content type
                    content_type = response.headers.get('content-type', '')
                    if not content_type.startswith('image/'):
                        print(f"Image verification failed: Invalid content type {content_type} for {image_url}")
                        return False

                    # Verify image data
                    image_data = await response.read()

                    # Reduced minimum file size to 2KB
                    if len(image_data) < 2000:
                        print(f"Image verification failed: File too small ({len(image_data)} bytes) for {image_url}")
                        return False

                    # For smaller files (between 2KB and 5KB), we'll still accept them but log a warning
                    if len(image_data) < 5000:
                        print(f"Image has smaller than ideal file size ({len(image_data)} bytes) but will be accepted: {image_url}")

                    img = Image.open(io.BytesIO(image_data))

                    # Reduced minimum resolution to 150px
                    if img.size[0] < 150 or img.size[1] < 150:
                        print(f"Image verification failed: Low resolution ({img.size}) for {image_url}")
                        return False
                    
                    # For smaller images (between 150px and 300px), we'll still accept them but log a warning
                    if img.size[0] < 300 or img.size[1] < 300:
                        print(f"Image has lower than ideal resolution ({img.size}) but will be accepted: {image_url}")

                    # More lenient aspect ratio check
                    aspect_ratio = img.size[0] / img.size[1]
                    if aspect_ratio < 0.2 or aspect_ratio > 5.0:
                        print(f"Image verification failed: Extreme aspect ratio ({aspect_ratio}) for {image_url}")
                        return False

                    # Skip color variance check as it may be too strict
                    print(f"Image verification passed for {image_url}")
                    return True

            except aiohttp.ClientError as e:
                print(f"Image verification failed: Connection error for {image_url}: {e}")
                return False

    except Exception as e:
        print(f"Image verification failed: Unexpected error for {image_url}: {e}")
        return False

async def optimize_image(image_data: bytes, max_size: int = 1600) -> Optional[bytes]:
    """Optimize image for web use while maintaining high quality"""
    try:
        if not image_data or len(image_data) < 100:
            print(f"Cannot optimize image: Empty or too small data ({len(image_data) if image_data else 0} bytes)")
            return None

        img = Image.open(io.BytesIO(image_data))

        # Get original size for logging
        original_size = img.size
        original_mode = img.mode

        # Convert to RGB if needed
        if img.mode != 'RGB':
            print(f"Converting image from {img.mode} to RGB")
            img = img.convert('RGB')

        # Resize if too large, but maintain higher resolution (1600px)
        if img.size[0] > max_size or img.size[1] > max_size:
            # Preserve aspect ratio
            ratio = max_size / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            print(f"Resizing image from {original_size} to {new_size}")
            # Use high quality downsampling
            img = img.resize(new_size, Image.LANCZOS)

        # Try different quality settings to find optimal balance
        quality_options = [95, 90, 85, 80]
        optimized_data = None
        best_size = float('inf')

        for quality in quality_options:
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=quality, optimize=True)
            current_data = output.getvalue()
            current_size = len(current_data)

            # If this is smaller than our best so far, keep it
            if current_size < best_size:
                best_size = current_size
                optimized_data = current_data

            # If we're already smaller than original, we can stop
            if best_size <= len(image_data):
                break

        # If all optimization attempts resulted in larger files, use original
        if best_size > len(image_data):
            print("Optimization would increase file size, using original image data")
            optimized_data = image_data

        # Log optimization results
        original_kb = len(image_data) / 1024
        optimized_kb = len(optimized_data) / 1024
        reduction = (1 - (optimized_kb / original_kb)) * 100 if original_kb > 0 else 0
        print(f"Optimized image: {original_kb:.1f}KB → {optimized_kb:.1f}KB ({reduction:.1f}% reduction)")

        return optimized_data

    except Exception as e:
        print(f"Error optimizing image: {e}")
        # Try a more basic approach as fallback
        try:
            # Just convert to JPEG with decent quality
            img = Image.open(io.BytesIO(image_data))
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Resize if extremely large
            if img.size[0] > 2500 or img.size[1] > 2500:
                ratio = 2500 / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.LANCZOS)

            # Try different quality settings
            for quality in [90, 85, 80]:
                output = io.BytesIO()
                img.save(output, format='JPEG', quality=quality)
                optimized_data = output.getvalue()

                # If smaller than original, use it
                if len(optimized_data) <= len(image_data):
                    print(f"Used fallback image optimization with quality {quality}")
                    return optimized_data

            # If all optimization attempts resulted in larger files, use original
            print("Fallback optimization would increase file size, using original")
            return image_data

        except Exception as fallback_error:
            print(f"Fallback optimization also failed: {fallback_error}")
            # Return original data as last resort
            return image_data
