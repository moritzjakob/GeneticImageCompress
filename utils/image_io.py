from PIL import Image, ImageDraw, ImageFont
import numpy as np

def load_image(path):
    img = Image.open(path).convert("RGB")
    img_array = np.array(img)

    #Extract number of different color values
    unique_colors = np.unique(img_array.reshape(-1, 3), axis=0)
    print(f"Loaded image has {len(unique_colors)} unique colors.")

    return img_array


def save_image(array, path):
    img = Image.fromarray(array.astype(np.uint8))
    img.save(path)


def create_palette_image(palette, color_image_size=20, max_cols=8, font_size=12,
                         fitness_value=None, image_name=None,
                         fitness_type=None, mutation_name=None, crossover_name=None):
    """
    Creates an image showing the color palette as an image with the correspoding RGB values.
    Adds metadata as an header text.
    """

    #Setup header 
    header_font_size = font_size + 4  
    try:
        font = ImageFont.truetype("arial.ttf", header_font_size)
    except IOError:
        font = ImageFont.load_default()

    line1 = f"{image_name} | Fitness: {fitness_value}" if fitness_value and image_name else ""

    components = []
    if fitness_type:
        components.append(f"Fitness: {fitness_type}")
    if mutation_name:
        components.append(f"Mutation: {mutation_name}")
    if crossover_name:
        components.append(f"Crossover: {crossover_name}")
    line2 = " | ".join(components)

    # Estimate image size
    num_colors = len(palette)
    cols = max_cols
    rows = (num_colors + cols - 1) // cols  

    line_spacing = 5
    line1_height = header_font_size + line_spacing
    line2_height = font_size + line_spacing
    empty_line_height = 10
    header_height = line1_height + line2_height + empty_line_height

    text_area_width = 100  #Space for RGB text

    img_width = cols * (color_image_size + text_area_width)
    img_height = header_height + rows * color_image_size

    img = Image.new("RGB", (img_width, img_height), "white")
    draw = ImageDraw.Draw(img)

    # Draw header 
    draw.text((10, 5), line1, fill="black", font=font)
    draw.text((10, line1_height), line2, fill="black", font=font)

    # Draw color with rgb values
    for i, color in enumerate(palette):
        row = i // cols
        col = i % cols

        x = col * (color_image_size + text_area_width)
        y = header_height + row * color_image_size

        color_tuple = tuple(int(c) for c in color)

        color_image = Image.new("RGB", (color_image_size, color_image_size), color_tuple)
        img.paste(color_image, (x, y))

        rgb_text = f"{color_tuple}"
        draw.text((x + color_image_size + 5, y + (color_image_size - font_size) // 2),
                  rgb_text, fill="black", font=font)

    return img

