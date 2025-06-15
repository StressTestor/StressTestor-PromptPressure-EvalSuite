"""
Generate a professional logo for PromptPressure Eval Suite.
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_logo():
    # Create assets directory if it doesn't exist
    os.makedirs("../assets", exist_ok=True)
    
    # Create a new image with a dark background
    width, height = 800, 400
    background_color = (25, 28, 36)
    primary_color = (100, 149, 237)  # Cornflower blue
    secondary_color = (255, 255, 255)  # White
    
    # Create image and drawing context
    image = Image.new('RGB', (width, height), background_color)
    draw = ImageDraw.Draw(image)
    
    try:
        # Try to use a professional font
        font_large = ImageFont.truetype("arialbd.ttf", 60)
        font_small = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        # Fallback to default font
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Draw the main text
    text = "PromptPressure"
    _, _, w, h = draw.textbbox((0, 0), text, font=font_large)
    x = (width - w) // 2
    y = (height - h) // 2 - 30
    
    # Draw text with a subtle shadow
    draw.text((x+2, y+2), text, fill=(0, 0, 0, 128), font=font_large)
    draw.text((x, y), text, fill=primary_color, font=font_large)
    
    # Draw subtitle
    subtitle = "EVAL SUITE"
    _, _, w, h = draw.textbbox((0, 0), subtitle, font=font_small)
    x = (width - w) // 2
    y = (height - h) // 2 + 40
    draw.text((x, y), subtitle, fill=secondary_color, font=font_small)
    
    # Save the logo
    image.save("../assets/logo.png", "PNG")
    print("Logo generated at assets/logo.png")

if __name__ == "__main__":
    create_logo()
