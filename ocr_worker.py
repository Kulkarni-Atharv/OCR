import cv2
import numpy as np
import logging

try:
    import easyocr
except ImportError:
    logging.warning("EasyOCR is not installed. Run: pip install easyocr")
    easyocr = None

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OCRWorker:
    """
    A worker class to handle text extraction using EasyOCR.
    Specifically optimized for detecting micro-level fonts captured by a global shutter camera.
    EasyOCR is chosen for its stable ARM64/Raspberry Pi support via PyTorch.
    """
    def __init__(self, lang='en', use_gpu=False):
        self.lang = lang
        self.use_gpu = use_gpu
        self.ocr_engine = None
        self._initialize_engine()

    def _initialize_engine(self):
        """Initializes the EasyOCR engine. First run will download the model (~100MB)."""
        if easyocr is None:
            logging.error("Cannot initialize OCR engine: EasyOCR library is missing.")
            return

        logging.info(f"Initializing EasyOCR (GPU: {self.use_gpu}, Lang: {self.lang})...")
        logging.info("Note: First run will download the model (~100MB). Please wait...")
        try:
            # gpu=False for Raspberry Pi CPU inference
            self.ocr_engine = easyocr.Reader([self.lang], gpu=self.use_gpu)
            logging.info("EasyOCR initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize EasyOCR: {e}")

    def preprocess_image(self, image, scale_factor=2.0):
        """
        Preprocesses the image to enhance micro-level fonts.
        1. Upscales the image (crucial for tiny text).
        2. Converts to grayscale.
        3. Applies adaptive thresholding for high contrast.
        """
        if image is None:
            return None

        # 1. Upscale — makes micro-fonts large enough for the AI model to read
        height, width = image.shape[:2]
        new_size = (int(width * scale_factor), int(height * scale_factor))
        upscaled = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)

        # 2. Convert to grayscale
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)

        # 3. Adaptive threshold — makes text crisp black-on-white
        processed = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        return processed  # EasyOCR accepts single-channel grayscale images

    def extract_text(self, image, preprocess=True):
        """
        Extracts text from the provided image frame.

        Args:
            image:      numpy array — the BGR image frame from OpenCV/Camera
            preprocess: bool — whether to apply micro-font preprocessing

        Returns:
            str: The extracted text combined into a single string.
        """
        if self.ocr_engine is None:
            logging.error("OCR engine is not loaded. Cannot extract text.")
            return ""

        if image is None:
            logging.warning("Provided image is None. Skipping OCR.")
            return ""

        # Apply preprocessing if requested
        process_img = self.preprocess_image(image) if preprocess else image

        logging.info("Starting text extraction...")
        try:
            # detail=1 returns bounding boxes + text + confidence
            # paragraph=False gives per-word/line results which is easier to debug
            results = self.ocr_engine.readtext(process_img, detail=1, paragraph=False)

            extracted_text = []
            for (bbox, text, confidence) in results:
                extracted_text.append(text)
                logging.info(f"Detected: '{text}' (Confidence: {confidence:.2f})")

            final_text = "\n".join(extracted_text)

            if not final_text:
                logging.info("No text detected in the image.")
            else:
                print("\n--- OCR EXTRACTION RESULT ---")
                print(final_text)
                print("-----------------------------\n")

            return final_text

        except Exception as e:
            logging.error(f"Error during OCR extraction: {e}")
            return ""
