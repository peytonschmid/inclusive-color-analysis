# seasonal_palettes.py
# Defines color palettes for the 12 seasonal categories in Seasonal Color Analysis
# Each class contains an array of 10 RGB tuples

def hex_to_rgb(hex_code: str) -> tuple[int, int, int]:
    """Convert hex string (e.g., '#FF5733') to an RGB tuple."""
    hex_code = hex_code.lstrip("#")
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

class BrightSpring:
    name = "Bright Spring"
    palette = [hex_to_rgb(c) for c in [
        "#FF6F61",  # coral red
        "#FFA07A",  # light salmon
        "#FFD700",  # bright gold
        "#ADFF2F",  # green yellow (chartreuse)
        "#00FA9A",  # medium spring green
        "#40E0D0",  # turquoise
        "#1E90FF",  # dodger blue
        "#BA55D3",  # medium orchid (purple)
        "#FF69B4",  # hot pink
        "#FF4500",  # orange red
    ]]


class TrueSpring:
    name = "True Spring"
    palette = [hex_to_rgb(c) for c in [
        "#FFA500",  # orange
        "#FFD700",  # gold
        "#FF7F50",  # coral
        "#FFB347",  # pastel orange
        "#FFE135",  # banana yellow
        "#98FB98",  # pale green
        "#40E0D0",  # turquoise
        "#87CEEB",  # sky blue
        "#FF69B4",  # hot pink
        "#CD5C5C",  # indian red
    ]]


class LightSpring:
    name = "Light Spring"
    palette = [hex_to_rgb(c) for c in [
        "#FFFACD",  # lemon chiffon
        "#FFDAB9",  # peach puff
        "#FAFAD2",  # light goldenrod yellow
        "#E0FFFF",  # light cyan
        "#E6E6FA",  # lavender
        "#FFB6C1",  # light pink
        "#FFE4E1",  # misty rose
        "#F5DEB3",  # wheat
        "#F0E68C",  # khaki
        "#B0E0E6",  # powder blue
    ]]


class LightSummer:
    name = "Light Summer"
    palette = [hex_to_rgb(c) for c in [
        "#B0E0E6",  # powder blue
        "#AFEEEE",  # pale turquoise
        "#E6E6FA",  # lavender
        "#D8BFD8",  # thistle
        "#F08080",  # light coral
        "#F5DEB3",  # wheat
        "#FFB6C1",  # light pink
        "#87CEFA",  # light sky blue
        "#D3D3D3",  # light gray
        "#F0FFF0",  # honeydew
    ]]


class TrueSummer:
    name = "True Summer"
    palette = [hex_to_rgb(c) for c in [
        "#4682B4",  # steel blue
        "#5F9EA0",  # cadet blue
        "#708090",  # slate gray
        "#6A5ACD",  # slate blue
        "#9370DB",  # medium purple
        "#DB7093",  # pale violet red
        "#C0C0C0",  # silver
        "#778899",  # light slate gray
        "#4169E1",  # royal blue
        "#8FBC8F",  # dark sea green
    ]]


class SoftSummer:
    name = "Soft Summer"
    palette = [hex_to_rgb(c) for c in [
        "#D8BFD8",  # thistle
        "#E0B0FF",  # mauve/lilac
        "#C3B1E1",  # light lavender
        "#DCDCDC",  # gainsboro (soft gray)
        "#B0C4DE",  # light steel blue
        "#C1CDC1",  # tea green / gray green
        "#E6E6FA",  # lavender
        "#AFEEEE",  # pale turquoise
        "#DDA0DD",  # plum
        "#BEBEBE",  # gray
    ]]

class SoftAutumn:
    name = "Soft Autumn"
    palette = [hex_to_rgb(c) for c in [
        "#C9A27E",  # soft camel
        "#DAB88B",  # wheat
        "#E3C565",  # muted mustard
        "#B59F3B",  # olive gold
        "#8E9A6C",  # sage/olive
        "#6E8B74",  # soft moss
        "#A77E6B",  # dusty terracotta
        "#C27D6A",  # softened coral clay
        "#8AA39B",  # muted teal
        "#7A6A8E",  # dusty plum
    ]]
    
class TrueAutumn:
    name = "True Autumn"
    palette = [hex_to_rgb(c) for c in [
        "#C7773D",  # pumpkin
        "#E0892E",  # squash orange
        "#C49A00",  # curry/mustard
        "#8B6B2E",  # caramel
        "#7A8B2E",  # olive green
        "#3E6B47",  # forest green
        "#0F766E",  # teal (warm-leaning)
        "#2AB7CA",  # warm turquoise accent
        "#B5544D",  # paprika
        "#9A4D82",  # warm plum
    ]]

class DeepAutumn:
    name = "Deep Autumn"
    palette = [hex_to_rgb(c) for c in [
        "#7A3B1A",  # warm chocolate brown
        "#9C3D18",  # russet / burnt sienna
        "#B1470E",  # terracotta / burnt orange
        # "#8C6A00",  # golden mustard
        "#996515",  # deep golden caramel
        "#556B2F",  # deep olive green
        "#3A5A40",  # forest green
        "#2E6E60",  # deep turquoise
        "#5E2B3A",  # deep wine / burgundy
        "#6E3A2C",  # brick red / auburn
        "#4A3A2C",  # chocolate taupe / deep neutral
    ]]

class DeepWinter:
    name = "Deep Winter"
    palette = [hex_to_rgb(c) for c in [
        "#000000",  # black (ultimate contrast anchor)
        "#191970",  # midnight navy
        "#006A4E",  # emerald green (cool, clear)
        "#4B0082",  # indigo / royal purple
        "#8B0000",  # dark crimson / burgundy
        "#FF0000",  # true red (blue-based)
        "#FF1493",  # icy pink (frosted highlight)
        "#2E0854",  # deep plum / aubergine
        "#009999",  # dark cyan / turquoise
        "#8B008B",  # deep fuchsia / magenta
    ]]


class TrueWinter:
    name = "True Winter"
    palette = [hex_to_rgb(c) for c in [
        "#4169E1",  # royal blue
        "#0000FF",  # pure blue
        "#8A2BE2",  # blue violet
        "#20B2AA",  # light sea green
        "#00CED1",  # dark turquoise
        "#008080",  # teal
        "#DC143C",  # crimson
        "#FF0000",  # red
        "#808080",  # gray
        "#000000",  # black
    ]]


class BrightWinter:
    name = "Bright Winter"
    palette = [hex_to_rgb(c) for c in [
        "#FF1493",  # deep fuchsia
        "#DC143C",  # crimson red
        "#FF69B4",  # hot pink
        "#FF0000",  # pure red
        "#00CED1",  # dark turquoise
        "#1E90FF",  # dodger blue
        "#00BFFF",  # deep sky blue
        "#7B68EE",  # medium slate blue
        "#32CD32",  # bright lime green
        "#FFFF00",  # vivid yellow
    ]]
