from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image #PIL (Pillow) → loads images easily in Python.
import torch #torch → PyTorch backend used by BLIP-2 to run computations.

class VisionLanguageModel:
    def __init__(self):
        print("🔄 Loading BLIP-2 Vision-Language Model...")

        # Use Apple GPU (MPS) if available
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"⚙️ Using device: {self.device}")

        # Load processor and model
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device)

        print("✅ Model loaded successfully!")

    def describe_image(self, image_path: str) -> str:
        """Takes an image path and returns a text description."""
        image = Image.open(image_path).convert("RGB")

        # Preprocess image for model
        inputs = self.processor(image, return_tensors="pt").to(self.device)

        # Generate a caption
        output = self.model.generate(**inputs, max_length=50)

        # Decode and return the caption
        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return caption
