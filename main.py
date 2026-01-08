import os
import sys
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont, ImageChops, ImageOps
from rembg import remove, new_session
from sklearn.cluster import KMeans
import gc  # For memory cleanup

# ---------------------------------------------------------
# CONFIGURATION & TUNING
# ---------------------------------------------------------

# LOCAL TESTING ONLY - These are IGNORED when imported by tele_image_editor.py!
# When running on Railway, the paths come from function arguments instead.
INPUT_FILENAME = "input.jpg"      # Only used when running: python main.py
OUTPUT_FILENAME = "viral_result.jpg"  # Only used when running: python main.py  
TEXT_OVERLAY = "WINNER"            # Only used when running: python main.py

# Canvas Settings
TARGET_SIZE = (1080, 1350)         # 4:5 Aspect Ratio (IG Portrait)

# Typography Settings
FONT_NAME = "Impact.ttf"           # Ensure this file exists
TEXT_PADDING_RATIO = 0.02          # 0.02 = 2% padding on left/right (Very tight)
TEXT_TOP_OFFSET_RATIO = 0.02       # 0.05 = Start text 5% down from the top edge
TEXT_GAP_RATIO = 0.15             # Gap between rows relative to font size. 
                                   # Negative values (e.g., -0.15) pull rows closer (tight stack).
                                   # Positive values add space.

# Grain/Texture Settings
GRAIN_INTENSITY = 0.15             # Opacity of the grain (0.0 to 1.0). 0.15 is subtle but visible.
GRAIN_ROUGHNESS = 50               # Variance of the noise. Higher = Chunkier/Rougher grain.
GRAIN_SIZE = 2                    # Grain size multiplier. 1 = fine, 2-3 = chunky/retro look

# Blur Settings
BLUR_INTENSITY = 5                # Gaussian blur radius for background. Higher = more blur

# Camera Filter Settings (Cinematic Color Grade)
CAMERA_FILTER_ENABLED = True       # Toggle the filter on/off
FILTER_WARMTH = 1.1                # >1 = warmer (orange tint), <1 = cooler (blue tint)
FILTER_SATURATION = 1.15           # Color saturation boost (1.0 = normal)
FILTER_CONTRAST = 1.08             # Contrast adjustment (1.0 = normal)
FILTER_SHADOWS_TINT = (10, 5, 20)  # RGB tint added to shadows (purple/blue for cinematic)
FILTER_HIGHLIGHTS_TINT = (15, 10, 0)  # RGB tint added to highlights (warm gold)

# Sharpness Settings (Unsharp Mask)
SHARPNESS_ENABLED = True           # Toggle sharpening on/off
SHARPNESS_RADIUS = 2               # Blur radius for unsharp mask (1-5 typical)
SHARPNESS_PERCENT = 100            # Sharpening strength (50-200 typical)
SHARPNESS_THRESHOLD = 3            # Pixel difference threshold (0-10 typical)

# Subject Positioning Settings
SMART_SUBJECT_POSITIONING = True   # Enable intelligent subject placement
SUBJECT_MIN_TOP_PERCENT = 0.20     # Subject's top must be at least 30% down from top (0.30 = 30%)

class ThumbnailGenerator:
    # Use lighter model (u2netp: 4MB vs u2net: 176MB) - shared across instances
    _rembg_session = None
    
    @classmethod
    def _get_rembg_session(cls):
        """Get or create shared rembg session with lightweight model."""
        if cls._rembg_session is None:
            print("      → Loading AI model (u2netp - lightweight)...")
            cls._rembg_session = new_session("u2netp")  # 4MB vs 176MB!
        return cls._rembg_session
    
    def __init__(self):
        self.width, self.height = TARGET_SIZE

    def _load_maximized_font(self, text, target_width):
        """
        Finds the largest possible font size that fits within the target width.
        """
        font_size = 600  # Start massive
        font = None
        
        # Fallback check
        try:
            ImageFont.truetype(FONT_NAME, 20)
        except OSError:
            print(f"[Warning] '{FONT_NAME}' not found. Using default.")
            return ImageFont.load_default(), 100

        # Iteratively reduce size until it fits
        while font_size > 20:
            font = ImageFont.truetype(FONT_NAME, font_size)
            bbox = font.getbbox(text)
            text_actual_width = bbox[2] - bbox[0]
            
            if text_actual_width <= target_width:
                return font, font_size, bbox
            
            font_size -= 5 # Decrement step
            
        return font, font_size, font.getbbox(text)

    def _get_dominant_color(self, image):
        """ Extracts dominant color using K-Means. """
        small_img = image.copy()
        small_img.thumbnail((150, 150))
        img_array = np.array(small_img)
        if len(img_array.shape) > 2:
            img_array = img_array.reshape((img_array.shape[0] * img_array.shape[1], 3))
        
        kmeans = KMeans(n_clusters=1, n_init=10)
        kmeans.fit(img_array)
        return tuple(kmeans.cluster_centers_[0].astype(int))

    def _get_subject_bounds(self, subject_rgba):
        """
        Analyzes a subject layer (RGBA with transparency) and returns the bounding box
        of non-transparent pixels.
        Returns: (top, bottom, left, right) pixel coordinates, or None if no subject found.
        """
        alpha = np.array(subject_rgba.split()[-1])  # Get alpha channel
        
        # Find rows and columns with non-transparent pixels
        rows_with_subject = np.any(alpha > 10, axis=1)
        cols_with_subject = np.any(alpha > 10, axis=0)
        
        if not np.any(rows_with_subject):
            return None  # No subject found
        
        top = np.argmax(rows_with_subject)
        bottom = len(rows_with_subject) - np.argmax(rows_with_subject[::-1]) - 1
        left = np.argmax(cols_with_subject)
        right = len(cols_with_subject) - np.argmax(cols_with_subject[::-1]) - 1
        
        return (top, bottom, left, right)

    def _smart_crop_and_position(self, original, text):
        """
        Intelligently crops the image to ensure subject doesn't occlude text.
        NEW APPROACH: Extend image height FIRST, then crop to 4:5.
        Returns: (base_img, subject_layer) - both at TARGET_SIZE
        """
        orig_w, orig_h = original.size
        target_w, target_h = TARGET_SIZE
        target_ratio = target_w / target_h
        
        # MEMORY OPTIMIZATION: Limit original image size for processing
        max_dimension = 2000  # Reduce if still OOM
        if max(orig_w, orig_h) > max_dimension:
            scale = max_dimension / max(orig_w, orig_h)
            original = original.resize((int(orig_w * scale), int(orig_h * scale)), Image.Resampling.LANCZOS)
            orig_w, orig_h = original.size
            print(f"      → Resized input to {orig_w}x{orig_h} for memory")
        
        # Simple fixed percentage: subject top must be at least X% down from top
        min_clear_zone = target_h * SUBJECT_MIN_TOP_PERCENT
        
        print(f"      → Min subject top: {int(min_clear_zone)}px ({int(SUBJECT_MIN_TOP_PERCENT * 100)}% from top)")
        
        # First, do a preliminary subject extraction on original to find the subject
        print("      → Analyzing subject position...")
        
        # MEMORY OPTIMIZATION: Scale down MORE aggressively for analysis (500px instead of 1000px)
        analysis_scale = min(1.0, 500 / max(orig_w, orig_h))
        analysis_size = (int(orig_w * analysis_scale), int(orig_h * analysis_scale))
        analysis_img = original.resize(analysis_size, Image.Resampling.LANCZOS)
        analysis_subject = remove(analysis_img, session=self._get_rembg_session())
        
        # Free memory immediately
        del analysis_img
        gc.collect()
        
        bounds = self._get_subject_bounds(analysis_subject)
        del analysis_subject
        gc.collect()
        
        if bounds is None:
            # No subject detected, use center crop
            print("      → No subject detected, using center crop")
            base_img = ImageOps.fit(original, TARGET_SIZE, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
            del original
            gc.collect()
            subject_layer = remove(base_img, session=self._get_rembg_session())
            return base_img, subject_layer
        
        # Scale bounds back to original size
        subject_top = int(bounds[0] / analysis_scale)
        subject_bottom = int(bounds[1] / analysis_scale)
        subject_left = int(bounds[2] / analysis_scale)
        subject_right = int(bounds[3] / analysis_scale)
        subject_center_x = (subject_left + subject_right) // 2
        
        # Calculate the FINAL crop width (full width for taller images, or calculated for wider)
        # We want to use full original width and calculate required height
        final_crop_w = orig_w
        final_crop_h = int(orig_w / target_ratio)  # Height needed for 4:5 ratio at full width
        
        # Calculate where subject top would be in final output
        # Scale factor from crop to target
        scale_factor = target_h / final_crop_h
        subject_top_in_target = subject_top * scale_factor
        
        # Calculate how much extra height we need at top
        pixels_needed_in_target = 0
        if subject_top_in_target < min_clear_zone:
            pixels_needed_in_target = int(min_clear_zone - subject_top_in_target)
            # Convert back to original image scale
            pixels_needed_in_original = int(pixels_needed_in_target / scale_factor)
            print(f"      → Need {pixels_needed_in_original}px extra height at top (extends to {pixels_needed_in_target}px in output)")
        else:
            pixels_needed_in_original = 0
            print("      → Subject already clear of text zone")
        
        # STEP 1: Extend original image by adding fill at TOP
        working_img = original
        if pixels_needed_in_original > 0:
            # Get dominant color for fill
            dom_color = self._get_dominant_color(original)
            
            # Create blurred version of top portion for seamless fill
            blur_sample = original.crop((0, 0, orig_w, min(200, orig_h)))
            blur_sample = blur_sample.resize((orig_w, pixels_needed_in_original), Image.Resampling.LANCZOS)
            blur_sample = blur_sample.filter(ImageFilter.GaussianBlur(radius=30))
            
            # Tint the blur to match scene
            tint = Image.new("RGBA", (orig_w, pixels_needed_in_original), dom_color + (120,))
            blur_sample = Image.alpha_composite(blur_sample.convert("RGBA"), tint).convert("RGB")
            
            # Create extended canvas
            new_height = orig_h + pixels_needed_in_original
            extended = Image.new("RGB", (orig_w, new_height), dom_color)
            
            # Paste blurred fill at top
            extended.paste(blur_sample, (0, 0))
            
            # Paste original below
            extended.paste(original, (0, pixels_needed_in_original))
            
            working_img = extended
            print(f"      → Extended image from {orig_h}px to {new_height}px height")
        
        # STEP 2: Now crop to 4:5 ratio from the extended image
        work_w, work_h = working_img.size
        work_ratio = work_w / work_h
        
        if work_ratio > target_ratio:
            # Image is wider than 4:5 - crop sides, center on subject
            crop_h = work_h
            crop_w = int(work_h * target_ratio)
            
            # Center crop on subject's horizontal position
            # But account for the extension - subject_center_x is still valid
            left = subject_center_x - crop_w // 2
            
            # Clamp to bounds
            if left < 0:
                left = 0
            elif left + crop_w > work_w:
                left = work_w - crop_w
            
            right = left + crop_w
            top = 0
            bottom = crop_h
            
            print(f"      → Centering subject horizontally (x={subject_center_x})")
        else:
            # Image is taller than or equal to 4:5 - crop top/bottom, use full width
            crop_w = work_w
            crop_h = int(work_w / target_ratio)
            
            # Align to TOP (to include our added fill)
            left = 0
            right = crop_w
            top = 0
            bottom = min(crop_h, work_h)
        
        # Crop and resize to target
        cropped = working_img.crop((left, top, right, bottom))
        del working_img
        gc.collect()
        
        base_img = cropped.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        del cropped
        gc.collect()
        
        # Extract subject from final result
        subject_layer = remove(base_img, session=self._get_rembg_session())
        gc.collect()
        
        return base_img, subject_layer


    def _add_film_grain(self, image):
        """ Generates and blends film grain noise onto the image. """
        # Calculate grain dimensions based on size multiplier
        # Larger GRAIN_SIZE = fewer noise pixels = chunkier grain
        grain_h = max(1, self.height // GRAIN_SIZE)
        grain_w = max(1, self.width // GRAIN_SIZE)
        
        # Create a numpy array of random noise centered around grey (128)
        sigma = GRAIN_ROUGHNESS
        noise = np.random.normal(128, sigma, (grain_h, grain_w, 3)).astype(np.uint8)
        
        # Create noise image and scale back up if grain size > 1
        noise_img = Image.fromarray(noise, "RGB")
        if GRAIN_SIZE > 1:
            noise_img = noise_img.resize(TARGET_SIZE, Image.Resampling.NEAREST)  # NEAREST keeps it chunky
        
        # Create a blend mask based on intensity
        mask = Image.new("L", TARGET_SIZE, int(255 * GRAIN_INTENSITY))
        
        # We overlay the noise onto the existing image
        # 'Overlay' blend mode preserves shadows/highlights better than simple transparency
        grain_layer = ImageChops.overlay(image.convert("RGB"), noise_img)
        
        # Blend original with grainy version using intensity mask
        return Image.composite(grain_layer, image.convert("RGB"), mask).convert("RGBA")

    def _apply_camera_filter(self, image):
        """ Applies a cinematic color grade filter for that viral Instagram look. """
        if not CAMERA_FILTER_ENABLED:
            return image
        
        img = image.convert("RGB")
        
        # 1. Apply Warmth (shift red/blue balance)
        img_array = np.array(img, dtype=np.float32)
        
        # Warm = boost reds, reduce blues
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] * FILTER_WARMTH, 0, 255)  # Red
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] / FILTER_WARMTH, 0, 255)  # Blue
        
        # 2. Apply Shadow/Highlight Tinting
        # Calculate luminance for each pixel
        luminance = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
        
        # Shadow mask (darker areas get more tint)
        shadow_mask = np.clip((128 - luminance) / 128, 0, 1)[:, :, np.newaxis]
        # Highlight mask (brighter areas get more tint)
        highlight_mask = np.clip((luminance - 128) / 128, 0, 1)[:, :, np.newaxis]
        
        # Apply tints
        for i, (s_tint, h_tint) in enumerate(zip(FILTER_SHADOWS_TINT, FILTER_HIGHLIGHTS_TINT)):
            img_array[:, :, i] += shadow_mask[:, :, 0] * s_tint
            img_array[:, :, i] += highlight_mask[:, :, 0] * h_tint
        
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array, "RGB")
        
        # 3. Apply Saturation
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(FILTER_SATURATION)
        
        # 4. Apply Contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(FILTER_CONTRAST)
        
        return img.convert("RGBA")

    def _create_gradient(self):
        """ Vertical black gradient for the footer. """
        gradient_height = int(self.height * 0.35)
        gradient_array = np.linspace(0, 255, gradient_height, dtype=np.uint8)
        gradient_array = (gradient_array / 255) ** 1.5 * 255 # Exponential fade
        gradient_array = gradient_array.astype(np.uint8)
        gradient_matrix = np.tile(gradient_array[:, np.newaxis], (1, self.width))
        alpha_mask = Image.fromarray(gradient_matrix, mode='L')
        black_gradient = Image.new("RGBA", (self.width, gradient_height), (0, 0, 0, 0))
        black_gradient.putalpha(alpha_mask)
        return black_gradient

    def create_thumbnail(self, image_path, text, output_path):
        print(f"--- Processing: {image_path} ---")

        # 1. LOAD & RESIZE
        try:
            original = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"❌ Error: {e}")
            return

        print("[1/6] Formatting Image...")
        
        # Use smart positioning if enabled, otherwise center crop
        if SMART_SUBJECT_POSITIONING:
            print("      → Smart subject positioning enabled")
            base_img, subject_layer = self._smart_crop_and_position(original, text)
            del original  # Free memory
            gc.collect()
        else:
            base_img = ImageOps.fit(original, TARGET_SIZE, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
            del original  # Free memory
            gc.collect()
            # 2. AI SUBJECT EXTRACTION
            print("[2/6] Extracting Subject (AI)...")
            subject_layer = remove(base_img, session=self._get_rembg_session())
            gc.collect()

        # 3. BACKGROUND TEXTURE (Blur + Tint + Grain)
        print("[3/6] Creating Textured Background...")
        blurred_bg = base_img.filter(ImageFilter.GaussianBlur(radius=BLUR_INTENSITY))
        
        # Tint
        dom_color = self._get_dominant_color(base_img)
        tint_layer = Image.new("RGBA", TARGET_SIZE, dom_color + (128,))
        bg_tinted = Image.alpha_composite(blurred_bg.convert("RGBA"), tint_layer)
        del blurred_bg, tint_layer  # Free memory
        gc.collect()
        
        # Add Grain
        bg_final = self._add_film_grain(bg_tinted)
        del bg_tinted  # Free memory
        gc.collect()

        # 4. TYPOGRAPHY ENGINE (Adjustable Stack)
        print("[4/6] Stacking Typography...")
        
        text_layer = Image.new("RGBA", TARGET_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_layer)
        
        # Calculate Max Width based on Padding Ratio
        allowed_width = self.width * (1.0 - (TEXT_PADDING_RATIO * 2))
        
        # Get Font
        font, font_size, bbox = self._load_maximized_font(text.upper(), allowed_width)
        
        # Calculate Vertical Positioning
        text_height = bbox[3] - bbox[1] # Actual height of letters
        font_top_offset = bbox[1]  # Font's internal top padding
        
        # Defines how much to move down per line
        # standard height + user defined gap
        step_y = text_height + (font_size * TEXT_GAP_RATIO) 
        
        # Start Y based on Offset config
        # Subtract font_top_offset to compensate for font's internal positioning
        current_y = (self.height * TEXT_TOP_OFFSET_RATIO) - font_top_offset 

        # Dynamic text rows - continue until opacity is 0 or we hit bottom
        # Custom opacity curve: 90% -> 60% -> 30% -> then -5% per row
        row_num = 0
        
        while current_y < self.height:
            # Calculate opacity based on row number
            if row_num == 0:
                opacity = int(255 * 0.90)  # 90%
            elif row_num == 1:
                opacity = int(255 * 0.60)  # 60%
            elif row_num == 2:
                opacity = int(255 * 0.30)  # 30%
            else:
                # After row 3, decrease by 5% (13 units) each time from 30%
                opacity = int(255 * 0.30) - ((row_num - 2) * int(255 * 0.05))
            
            if opacity <= 0:
                break
                
            row_text = text.upper()
            row_bbox = font.getbbox(row_text)
            row_width = row_bbox[2] - row_bbox[0]
            
            # Center X
            x_pos = (self.width - row_width) // 2
            
            # Draw Text
            draw.text((x_pos, current_y), row_text, font=font, fill=(255, 255, 255, opacity))
            
            # Increment Y for next row
            current_y += step_y
            row_num += 1

        # Blend Text (Overlay Mode)
        overlay_text = ImageChops.overlay(bg_final.convert("RGB"), text_layer.convert("RGB"))
        overlay_text = overlay_text.convert("RGBA")
        bg_final.paste(overlay_text, (0,0), text_layer)

        # 5. FINAL COMPOSITION
        print("[5/6] Compositing Layers...")
        comp = bg_final
        
        # Subject
        comp.alpha_composite(subject_layer)
        
        # Gradient
        gradient = self._create_gradient()
        comp.paste(gradient, (0, self.height - gradient.height), gradient)
        
        # 6. CAMERA FILTER & POLISH
        print("[6/6] Applying Camera Filter & Final Polish...")
        comp = self._apply_camera_filter(comp)
        
        # Final Polish
        final_image = comp.convert("RGB")
        
        # Apply Unsharp Mask for professional sharpening
        if SHARPNESS_ENABLED:
            final_image = final_image.filter(
                ImageFilter.UnsharpMask(
                    radius=SHARPNESS_RADIUS,
                    percent=SHARPNESS_PERCENT,
                    threshold=SHARPNESS_THRESHOLD
                )
            )

        final_image.save(output_path, quality=95)
        print(f"✅ Done! Saved to: {output_path}")

# ---------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    if not os.path.exists(INPUT_FILENAME):
        print(f"\n❌ Error: '{INPUT_FILENAME}' not found.")
        sys.exit(1)

    generator = ThumbnailGenerator()
    generator.create_thumbnail(INPUT_FILENAME, TEXT_OVERLAY, OUTPUT_FILENAME)
