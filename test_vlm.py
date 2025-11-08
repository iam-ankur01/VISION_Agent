from src.vlm_model import VisionLanguageModel

# Initialize the Vision-Language Model
vlm = VisionLanguageModel()

# Describe an image from assets/
image_path = "/Users/ankurmishra/Documents/BIGproject/VISION_AGENT/vision_agent/blond-hair-girl-taking-photo-260nw-2492842415.webp"

caption = vlm.describe_image(image_path)
print("🖼️ Image Description:", caption)
