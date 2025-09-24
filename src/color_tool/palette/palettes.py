# seasonal_palettes.py
# Defines color palettes for the 12 seasonal categories in Seasonal Color Analysis
# Each class contains an array of 10 RGB tuples

def hex_to_rgb(hex_code: str) -> tuple[int, int, int]:
    """Convert hex string (e.g., '#FF5733') to an RGB tuple."""
    hex_code = hex_code.lstrip("#")
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))


class BrightSpring:
    palette = [hex_to_rgb(c) for c in [
        "#FF6F61", "#FFA07A", "#FFD700", "#ADFF2F", "#00FA9A",
        "#40E0D0", "#1E90FF", "#9370DB", "#FF69B4", "#FF4500"
    ]]


class TrueSpring:
    palette = [hex_to_rgb(c) for c in [
        "#FFA500", "#FFD700", "#FF7F50", "#FFB347", "#FFE135",
        "#98FB98", "#40E0D0", "#87CEEB", "#FF69B4", "#CD5C5C"
    ]]


class LightSpring:
    palette = [hex_to_rgb(c) for c in [
        "#FFFACD", "#FFDAB9", "#FAFAD2", "#E0FFFF", "#E6E6FA",
        "#FFB6C1", "#FFE4E1", "#F5DEB3", "#F0E68C", "#B0E0E6"
    ]]


class LightSummer:
    palette = [hex_to_rgb(c) for c in [
        "#B0E0E6", "#AFEEEE", "#E6E6FA", "#D8BFD8", "#F08080",
        "#F5DEB3", "#FFB6C1", "#87CEFA", "#D3D3D3", "#F0FFF0"
    ]]


class TrueSummer:
    palette = [hex_to_rgb(c) for c in [
        "#4682B4", "#5F9EA0", "#708090", "#6A5ACD", "#9370DB",
        "#DB7093", "#C0C0C0", "#778899", "#4169E1", "#8FBC8F"
    ]]


class SoftSummer:
    palette = [hex_to_rgb(c) for c in [
        "#D8BFD8", "#E0B0FF", "#C3B1E1", "#DCDCDC", "#B0C4DE",
        "#C1CDC1", "#E6E6FA", "#AFEEEE", "#DDA0DD", "#BEBEBE"
    ]]


class SoftAutumn:
    palette = [hex_to_rgb(c) for c in [
        "#C19A6B", "#DEB887", "#D2B48C", "#BC8F8F", "#F4A460",
        "#E9967A", "#CD853F", "#DAA520", "#8B4513", "#B22222"
    ]]


class TrueAutumn:
    palette = [hex_to_rgb(c) for c in [
        "#8B4513", "#A0522D", "#D2691E", "#CD853F", "#DEB887",
        "#F4A460", "#B8860B", "#C19A6B", "#FF8C00", "#A52A2A"
    ]]


class DarkAutumn:
    palette = [hex_to_rgb(c) for c in [
        "#654321", "#8B4513", "#A0522D", "#B22222", "#800000",
        "#6B4226", "#B87333", "#964B00", "#556B2F", "#2F4F4F"
    ]]


class DarkWinter:
    palette = [hex_to_rgb(c) for c in [
        "#000000", "#191970", "#2F4F4F", "#4B0082", "#800000",
        "#8B0000", "#483D8B", "#2E0854", "#000080", "#8B008B"
    ]]


class TrueWinter:
    palette = [hex_to_rgb(c) for c in [
        "#4169E1", "#0000FF", "#8A2BE2", "#20B2AA", "#00CED1",
        "#008080", "#DC143C", "#FF0000", "#808080", "#000000"
    ]]


class BrightWinter:
    palette = [hex_to_rgb(c) for c in [
        "#FF1493", "#FF4500", "#FF69B4", "#FF6347", "#00CED1",
        "#00BFFF", "#1E90FF", "#7B68EE", "#32CD32", "#FFFF00"
    ]]
