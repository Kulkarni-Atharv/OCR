import cv2
import numpy as np
import logging
import os

# Prevent OpenBLAS thread crashes on ARM
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:
    logging.warning("RapidOCR is not installed. Run: pip install rapidocr-onnxruntime")
    RapidOCR = None

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OCRWorker:
    """
    Text extraction using PaddleOCR models via ONNX Runtime.
    Uses the same PP-OCRv3/v4 models as PaddleOCR but through ONNX Runtime
    which has stable ARM64 (Raspberry Pi) support.
    Optimized for detecting micro-level fonts from a global shutter camera.
    """
    def __init__(self, lang='en'):
        self.lang = lang
        self.ocr_engine = None
        self._initialize_engine()

    def _initialize_engine(self):
        """Initializes the RapidOCR engine (PaddleOCR models on ONNX Runtime)."""
        if RapidOCR is None:
            logging.error("Cannot initialize OCR engine: rapidocr-onnxruntime is missing.")
            return

        logging.info("Initializing PaddleOCR models via ONNX Runtime...")
        try:
            self.ocr_engine = RapidOCR()
            logging.info("OCR engine initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize OCR engine: {e}")

    def preprocess_image(self, image):
        """
        Preprocesses the image to enhance micro-level fonts.
        - Resizes to a safe resolution for Pi RAM.
        - Sharpens to make micro-fonts crisper.
        - Does NOT apply heavy thresholding (RapidOCR handles binarization internally).
        """
        if image is None:
            return None

        # Resize to a safe resolution — large enough for OCR, small enough for Pi RAM
        target_w, target_h = 1024, 768
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Sharpen to make micro-fonts crisper
        sharpen_kernel = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])
        sharpened = cv2.filter2D(gray, -1, sharpen_kernel)

        # Convert back to BGR (3-channel) as OCR expects color images
        processed_bgr = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

        return processed_bgr

    def extract_text(self, image, preprocess=True):
        """
        Extracts text from the provided image frame.

        Args:
            image:      numpy array — BGR image from OpenCV/Camera
            preprocess: bool — whether to apply micro-font preprocessing

        Returns:
            str: The extracted text as a single string.
        """
        if self.ocr_engine is None:
            logging.error("OCR engine is not loaded. Cannot extract text.")
            return ""

        if image is None:
            logging.warning("Provided image is None. Skipping OCR.")
            return ""

        process_img = self.preprocess_image(image) if preprocess else image

        logging.info("Starting text extraction...")
        try:
            # RapidOCR returns: (result, elapsed_time)
            # result is a list of [box, text, confidence] or None
            result, elapsed = self.ocr_engine(process_img)

            extracted_text = []
            if result:
                for line in result:
                    try:
                        text = line[1]
                        confidence = float(line[2])  # Cast to float — RapidOCR may return str
                        extracted_text.append(text)
                        logging.info(f"Detected: '{text}' (Confidence: {confidence:.2f})")
                    except (IndexError, TypeError, ValueError):
                        continue

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
