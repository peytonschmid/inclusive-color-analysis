import os
from typing import List, Tuple, Iterable, Union
from seasonal_palettes import BrightSpring, TrueSpring, LightSpring
from seasonal_palettes import LightSummer, TrueSummer, SoftSummer
from seasonal_palettes import SoftAutumn, TrueAutumn, DeepAutumn
from seasonal_palettes import DeepWinter, TrueWinter, BrightWinter
from PIL import Image, ImageDraw, ImageFont

PALETTES = [BrightSpring, TrueSpring, LightSpring, LightSummer, TrueSummer, SoftSummer, SoftAutumn, TrueAutumn, DeepAutumn, DeepWinter, TrueWinter, BrightWinter]

# Utility functions
def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Convert an RGB tuple to a hex color code."""
    r, g, b = rgb
    return f"#{r:02X}{g:02X}{b:02X}"

def _text_size(draw: ImageDraw.ImageDraw, text:str, font: ImageFont.ImageFont):
    """ 
    Robust text measurement. Uses textbbox (new), falls back to textsize if needed
    Returns (width, height)
    """
    if hasattr(draw, "textbbox"):
        # bbox = (left, top, right, bottom)
        left, top, right, bottom = draw.textbbox((0,0), text, font=font)
        return (right-left), (bottom-top)
    # fallback for older pillow 
    return draw.textsize(text, font=font)

# Palette generator class
class PaletteGenerator:
    def __init__(
        self,
        palette: Iterable[Tuple[int, int, int]],
        save_name: str = "example_palette.png",
        title: str = "Example Palette",
        columns: int = 4,
        swatch_size: int = 100,
        gap: int = 10,
        show_hex_labels: bool = True,
        bg_color: Tuple[int, int, int] = (240, 240, 240)
    ):
        """
        Initialize the PaletteGenerator with configuration options.

        Args:
            palette (Iterable[Tuple[int, int, int]]): List of colors in RGB format.
            save_name (str): Path to save the rendered palette image.
            title (str): Title for the palette image.
            columns (int): Number of columns in the palette grid.
            swatch_size (int): Size of each color swatch.
            gap (int): Gap between swatches.
            show_hex_labels (bool): Whether to display hex labels below swatches.
            bg_color (Tuple[int, int, int]): Background color of the palette.
        """
        self.palette = list(palette)  # Ensure the palette is a list of RGB tuples
        self.save_name = save_name
        self.title = title
        self.columns = columns
        self.swatch_size = swatch_size
        self.gap = gap
        self.show_hex_labels = show_hex_labels
        self.bg_color = bg_color

        imgs_path = os.path.join(os.path.dirname(__file__), "imgs")
        os.makedirs(imgs_path, exist_ok=True)  # Ensure the directory exists
        self.save_path = os.path.join(imgs_path, self.save_name)



    def generate_palette(self):
        """
        Generate the palette image and save it to the specified path.
        """
        print(f"Rendering palette with {len(self.palette)} colors.")
        save_path = self.create_image()
        print(f"Saving palette {self.save_name} to {save_path}.")
        print(f"Title: {self.title}")
        print(f"Columns: {self.columns}, Swatch Size: {self.swatch_size}, Gap: {self.gap}")
        print(f"Show Hex Labels: {self.show_hex_labels}, Background Color: {self.bg_color}")
        # Placeholder: Replace with actual rendering logic
        for color in self.palette:
            print(f"Color: {_rgb_to_hex(color)}")
        print("Palette generation complete.")

    def create_image(self):
        """
        Generate the palette image and save it to the specified path.
        """
        # --- fonts & metric ---
        font = ImageFont.load_default()
        title_font = ImageFont.truetype("src/color_tool/fonts/DancingScript-VariableFont_wght.ttf", size=48)

        # compute label size from font metrics 
        ascent, descent = font.getmetrics()
        label_h = (ascent + descent) if self.show_hex_labels else 0

        # compute title height from actual text height (if provided)
        title_h = 0 
        title_w = 0 
        if self.title: 
            dummy = Image.new("RGB", (1,1))
            d0 = ImageDraw.Draw(dummy)
            title_w, title_h = _text_size(d0, self.title, title_font)
            # add padding 
            title_h += 8
        
        # --- layout ---
        cell_w = self.swatch_size
        swatch_h = self.swatch_size
        cell_h = swatch_h + label_h

        n = len(self.palette)
        cols = max(1, self.columns)
        rows = (n + cols - 1) // cols

        width = self.gap + cols * (cell_w + self.gap)
        height = title_h + self.gap + rows * (cell_h + self.gap) + self.gap 

        img = Image.new("RGB", (width, height), self.bg_color)
        draw = ImageDraw.Draw(img)

        # --- draw title ---
        if self.title:
            # center title using width 
            tx = (width - title_w) // 2
            ty = (title_h - (_text_size(draw, self.title, title_font)[1])) / 2
            draw.text((tx, ty), self.title, fill=(30,30,30), font=title_font)

        # --- draw swatches ---
        for i, rgb in enumerate(self.palette):
            r,c = divmod(i, cols)
            x0 = self.gap + c * (cell_w + self.gap)
            y0 = title_h + self.gap + r * (cell_h + self.gap)
            x1 = x0 + cell_w
            y1 = y0 + swatch_h

            # draw swatch
            draw.rectangle([x0, y0, x1, y1], fill=rgb, outline=(220,220,220))

            # draw hex label
            if self.show_hex_labels:
                hex_color = _rgb_to_hex(rgb)
                tw, th = _text_size(draw, hex_color, font)
                tx = x0 + (cell_w - tw) // 2
                ty = y1 + 2
                draw.text((tx, ty), hex_color, fill=(30,30,30), font=font)

        # save image
        img.save(self.save_path)
        print(f"Palette saved to {self.save_path}")
        return self.save_path

        # Calculate the number of rows needed
        rows = (len(self.palette) + self.columns - 1) // self.columns

        # Calculate the image dimensions
        width = self.columns * self.swatch_size + (self.columns - 1) * self.gap
        height = rows * self.swatch_size + (rows - 1) * self.gap
        if self.title:
            height += self.swatch_size  # Add space for the title

        # Create the image
        image = Image.new("RGB", (width, height), self.bg_color)
        draw = ImageDraw.Draw(image)

        # Draw the title if specified
        if self.title:
            title_font = ImageFont.load_default()  # Use default font
            title_height = self.swatch_size // 2
            draw.text(
                (width // 2, title_height // 2),
                self.title,
                fill=(0, 0, 0),
                anchor="mm",
                font=title_font,
            )

        # Draw the color swatches
        for idx, color in enumerate(self.palette):
            row = idx // self.columns
            col = idx % self.columns

            x0 = col * (self.swatch_size + self.gap)
            y0 = row * (self.swatch_size + self.gap) + (self.swatch_size if self.title else 0)
            x1 = x0 + self.swatch_size
            y1 = y0 + self.swatch_size

            # Draw the swatch
            draw.rectangle([x0, y0, x1, y1], fill=color)

            # Draw the hex label if enabled
            if self.show_hex_labels:
                hex_color = _rgb_to_hex(color)
                label_font = ImageFont.load_default()  # Use default font
                text_width, text_height = draw.textsize(hex_color, font=label_font)
                text_x = x0 + (self.swatch_size - text_width) // 2
                text_y = y1 + 2  # Add a small gap below the swatch
                draw.text((text_x, text_y), hex_color, fill=(0, 0, 0), font=label_font)

        # Save the image
        image.save(self.save_path)
        print(f"Palette saved to {self.save_path}")

# Example usage
if __name__ == "__main__":
    # Use the BrightSpring palette from seasonal_palettes.py
    selected_palette = BrightSpring.palette

    for palette in PALETTES: 
        print(f"Palette: {palette.__name__}")

        # Create an instance of PaletteGenerator
        palette_generator = PaletteGenerator(
            palette=palette.palette,
            save_name=palette.name.replace(" ","_").lower() + "_palette.png",
            title=palette.name + " Palette",
            columns=5,
            swatch_size=120,
            gap=15,
            show_hex_labels=True,
            bg_color=(255, 255, 255),
        )

        # Generate the palette
        palette_generator.generate_palette()