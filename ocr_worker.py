import cv2
import numpy as np
import logging

try:
    from paddleocr import PaddleOCR
except ImportError:
    logging.warning("PaddleOCR is not installed. Please install it using: pip install paddlepaddle paddleocr")
    PaddleOCR = None

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OCRWorker:
    """
    A worker class to handle text extraction using PaddleOCR.
    Specifically optimized for detecting micro-level fonts captured by a global shutter camera.
    """
    def __init__(self, use_gpu=False, lang='en'):
        self.use_gpu = use_gpu
        self.lang = lang
        self.ocr_engine = None
        self._initialize_engine()

    def _initialize_engine(self):
        """Initializes the PaddleOCR engine."""
        if PaddleOCR is None:
            logging.error("Cannot initialize OCR engine: PaddleOCR library is missing.")
            return
            
        logging.info(f"Initializing PaddleOCR (GPU: {self.use_gpu}, Lang: {self.lang})...")
        try:
            # use_angle_cls=True helps if the paper is slightly rotated
            # Note: use_gpu was removed in PaddleOCR v3.x — GPU is auto-detected
            self.ocr_engine = PaddleOCR(use_angle_cls=True, lang=self.lang, show_log=False)
            logging.info("PaddleOCR initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize PaddleOCR: {e}")

    def preprocess_image(self, image, scale_factor=2.0):
        """
        Preprocesses the image to enhance micro-level fonts.
        1. Upscales the image.
        2. Converts to grayscale.
        3. Applies adaptive thresholding for better contrast.
        """
        if image is None:
            return None
            
        # 1. Upscale the image to make micro-fonts larger for the AI
        height, width = image.shape[:2]
        new_size = (int(width * scale_factor), int(height * scale_factor))
        upscaled = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
        
        # 2. Convert to Grayscale
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        
        # 3. Apply Thresholding (Optional, but helps with tiny text on paper)
        # We use a simple threshold or adaptive. Here we use adaptive Gaussian.
        processed = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Convert back to 3-channel as PaddleOCR expects RGB/BGR-like shape usually
        processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        
        return processed_bgr

    def extract_text(self, image, preprocess=True):
        """
        Extracts text from the provided image frame.
        
        Args:
            image: numpy array (the image frame from OpenCV/Camera)
            preprocess: boolean, whether to apply micro-font preprocessing
            
        Returns:
            str: The extracted text combined into a single string.
        """
        if self.ocr_engine is None:
            logging.error("OCR engine is not loaded.")
            return ""

        if image is None:
            logging.warning("Provided image is None. Skipping OCR.")
            return ""

        # Preprocess if requested
        process_img = self.preprocess_image(image) if preprocess else image

        logging.info("Starting text extraction...")
        try:
            # Run OCR
            result = self.ocr_engine.ocr(process_img, cls=True)
            
            extracted_text = []
            if result and result[0]:
                for line in result[0]:
                    # line format: [[box coordinates], (text, confidence)]
                    text = line[1][0]
                    confidence = line[1][1]
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
