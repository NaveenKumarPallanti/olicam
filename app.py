import streamlit as st
from PIL import Image, ImageOps, ImageFilter, ImageDraw
import numpy as np
from rembg import remove
import io

st.set_page_config(page_title="Product Studio Background Remover", layout="centered")

st.title("🛍️ Product Studio Background Studio")
st.write(
    "Snap or upload a photo of your product. The app will remove the messy "
    "background and drop it into a dark studio setup: soft overhead glow, "
    "a subtle floor reflection, and a grounded contact shadow."
)

# Cap the longest edge before processing. Phone cameras produce huge images
# (often 3000-4000px) which slow down segmentation a lot without improving
# the final result, since the studio backdrop is generated at this same size.
MAX_DIMENSION = 1600


def load_image_input():
    """Lets the user either upload a file or take a photo with their camera."""
    tab_upload, tab_camera = st.tabs(["📁 Upload a photo", "📷 Use camera"])

    with tab_upload:
        file_from_upload = st.file_uploader(
            "Choose a product image...", type=["jpg", "jpeg", "png"]
        )

    with tab_camera:
        file_from_camera = st.camera_input("Take a picture of your product")

    return file_from_upload or file_from_camera


def downscale_if_needed(img: Image.Image, max_dim: int = MAX_DIMENSION) -> Image.Image:
    width, height = img.size
    longest_edge = max(width, height)
    if longest_edge <= max_dim:
        return img
    scale = max_dim / longest_edge
    new_size = (int(width * scale), int(height * scale))
    return img.resize(new_size, Image.LANCZOS)


def create_studio_background(size, wall_color=(18, 18, 18), floor_color=(30, 30, 30)):
    """
    Dark studio backdrop: a radial glow on the 'wall' (upper ~68% of frame)
    fading from a lighter center-top to near-black edges, meeting a slightly
    lighter 'floor' band at the bottom that reads as a reflective surface.
    """
    width, height = size
    floor_line = int(height * 0.68)  # where wall meets floor

    Y, X = np.ogrid[:height, :width]

    # --- Wall: radial gradient centered above the product ---
    center_x, center_y = width // 2, int(height * 0.28)
    max_radius = np.hypot(width, height * 0.75) / 1.4
    dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    wall_gradient = np.clip(dist / max_radius, 0, 1)

    glow_boost = 1.6  # how much lighter the glow center is than wall_color
    wall = np.stack(
        [np.clip(c * (1 + (glow_boost - 1) * (1 - wall_gradient)), 0, 255) for c in wall_color],
        axis=-1,
    )

    # --- Floor: solid-ish band, darkening slightly toward the bottom edge (toward camera) ---
    floor_fade = np.clip((Y - floor_line) / max(height - floor_line, 1), 0, 1)
    floor = np.stack(
        [np.clip(c * (1 - 0.35 * floor_fade), 0, 255) for c in floor_color],
        axis=-1,
    )

    canvas = np.where(Y[..., None] < floor_line, wall, floor).astype(np.float32)

    # --- Vignette: darken corners so the eye stays on the product ---
    cx, cy = width / 2, height / 2
    vign_dist = np.sqrt(((X - cx) / (width / 2)) ** 2 + ((Y - cy) / (height / 2)) ** 2)
    vignette = np.clip(1 - 0.5 * np.clip(vign_dist - 0.6, 0, None), 0, 1)
    canvas = canvas * vignette[..., None]

    return Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), mode="RGB")


def make_contact_shadow(product_rgba: Image.Image) -> Image.Image:
    """
    Builds a flattened elliptical contact shadow beneath the product,
    rather than a blurred copy of the whole silhouette, so it reads as
    'sitting on a surface' instead of a halo around the product.
    Returns an 'L' mode mask the same size as the product image.
    """
    width, height = product_rgba.size
    alpha = np.array(product_rgba.split()[3])

    rows_with_content = np.where(alpha.max(axis=1) > 10)[0]
    if len(rows_with_content) == 0:
        return Image.new("L", product_rgba.size, 0)

    base_row = int(rows_with_content[-1])
    band_top = max(base_row - 25, 0)
    base_band = alpha[band_top:base_row + 1]
    cols_with_content = np.where(base_band.max(axis=0) > 10)[0]
    if len(cols_with_content) == 0:
        return Image.new("L", product_rgba.size, 0)

    left, right = int(cols_with_content.min()), int(cols_with_content.max())
    product_width = max(right - left, 1)
    center_x = (left + right) // 2

    shadow_layer = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(shadow_layer)

    ellipse_w = int(product_width * 0.9)
    ellipse_h = max(int(ellipse_w * 0.14), 10)

    draw.ellipse(
        [
            center_x - ellipse_w // 2,
            base_row - ellipse_h // 2,
            center_x + ellipse_w // 2,
            base_row + ellipse_h // 2,
        ],
        fill=160,
    )

    return shadow_layer.filter(ImageFilter.GaussianBlur(max(ellipse_h * 0.6, 4)))


def make_reflection(product_rgba: Image.Image, fade_strength: float = 0.35) -> Image.Image:
    """
    Flips the product vertically and fades it out quickly to suggest a faint
    reflection on a glossy studio floor. Returned image is the same size as
    the product image; the caller pastes it starting at the product's base row.
    """
    flipped = ImageOps.flip(product_rgba)
    r, g, b, a = flipped.split()
    a_array = np.array(a).astype(np.float32)

    height = a_array.shape[0]
    fade = np.linspace(fade_strength, 0, height).reshape(-1, 1)
    faded_alpha = Image.fromarray(np.clip(a_array * fade, 0, 255).astype(np.uint8))

    return Image.merge("RGBA", (r, g, b, faded_alpha))


def image_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


uploaded_file = load_image_input()

if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert("RGB")
    input_image = downscale_if_needed(input_image)

    st.subheader("Original Image")
    st.image(input_image, use_container_width=True)

    if st.button("✨ Process Studio Shot"):
        try:
            with st.spinner("Isolating product and rendering studio lighting..."):
                width, height = input_image.size

                # Step 1: Remove background
                product = remove(input_image)

                # Step 2: Studio backdrop (wall + floor + vignette)
                background = create_studio_background((width, height))

                # Step 3: Find where the product sits (its lowest opaque pixel)
                alpha_arr = np.array(product.split()[3])
                rows_with_content = np.where(alpha_arr.max(axis=1) > 10)[0]
                base_row = int(rows_with_content[-1]) if len(rows_with_content) else height - 1

                # Step 4: Contact shadow, positioned right at the product's base
                shadow_mask = make_contact_shadow(product)

                # Step 5: Faint floor reflection, anchored at the product's base
                reflection = make_reflection(product)

                # Step 6: Composite: background -> reflection -> shadow -> product
                final_image = background.copy()

                final_image.paste(reflection, (0, base_row), reflection)

                black_layer = Image.new("RGB", (width, height), (0, 0, 0))
                final_image.paste(black_layer, (0, 0), shadow_mask)

                final_image.paste(product, (0, 0), product)

            st.subheader("Studio Output")
            st.image(final_image, use_container_width=True)

            st.download_button(
                label="📥 Download Studio Image",
                data=image_to_bytes(final_image),
                file_name="studio_product.png",
                mime="image/png",
            )
        except Exception as e:
            st.error(f"Something went wrong while processing the image: {e}")