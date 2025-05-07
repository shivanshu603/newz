from PIL import Image, ImageDraw, ImageFont
import io
import base64
from config import Config
from utils import logger

class MultimediaGenerator:
    def __init__(self):
        self.image_width = 800
        self.image_height = 600
        self.font_path = "arial.ttf"

    def add_multimedia(self, content: str, topic: str) -> str:
        """Add multimedia elements to content"""
        try:
            # Generate and add images
            content = self._add_images(content, topic)
            
            # Generate and add infographics
            content = self._add_infographics(content, topic)
            
            return content
            
        except Exception as e:
            logger.error(f"Error adding multimedia: {str(e)}")
            return content

    def _add_images(self, content: str, topic: str) -> str:
        """Add relevant images to content"""
        # Generate placeholder images
        image_html = self._generate_image_html(topic)
        return f"{content}\n{image_html}"

    def _add_infographics(self, content: str, topic: str) -> str:
        """Add infographics to content"""
        # Generate placeholder infographics
        infographic_html = self._generate_infographic_html(topic)
        return f"{content}\n{infographic_html}"

    def _generate_image_html(self, topic: str) -> str:
        """Generate HTML for an image"""
        try:
            # Create a placeholder image
            img = Image.new('RGB', (self.image_width, self.image_height), color = (73, 109, 137))
            d = ImageDraw.Draw(img)
            font = ImageFont.load_default()
            d.text((10,10), f"Placeholder Image for: {topic}", fill=(255,255,0), font=font)
            
            # Convert to base64
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return f'<img src="data:image/png;base64,{img_str}" alt="{topic} image">'
            
        except Exception as e:
            logger.error(f"Error generating image: {str(e)}")
            return ""

    def _generate_infographic_html(self, topic: str) -> str:
        """Generate HTML for an infographic"""
        try:
            # Create a placeholder infographic
            img = Image.new('RGB', (self.image_width, self.image_height), color = (73, 109, 137))
            d = ImageDraw.Draw(img)
            font = ImageFont.load_default()
            d.text((10,10), f"Placeholder Infographic for: {topic}", fill=(255,255,0), font=font)
            
            # Convert to base64
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return f'<img src="data:image/png;base64,{img_str}" alt="{topic} infographic">'
            
        except Exception as e:
            logger.error(f"Error generating infographic: {str(e)}")
            return ""
