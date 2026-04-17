import cv2
import numpy as np
import logging
import os

# Fix segfault on ARM (Raspberry Pi) caused by OpenBLAS multi-threading conflicts
# Must be set BEFORE importing paddleocr
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

try:
    from paddleocr import PaddleOCR
except ImportError:
    logging.warning("PaddleOCR is not installed. Run: pip install paddlepaddle paddleocr")
    PaddleOCR = None

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OCRWorker:
    """
    A worker class to handle text extraction using PaddleOCR.
    Optimized for micro-level fonts on Raspberry Pi (ARM64).
    """
    def __init__(self, lang='en'):
        self.lang = lang
        self.ocr_engine = None
        self._initialize_engine()

    def _initialize_engine(self):
        """Initializes the PaddleOCR engine with lightweight settings for Raspberry Pi."""
        if PaddleOCR is None:
            logging.error("Cannot initialize OCR engine: PaddleOCR library is missing.")
            return

        logging.info(f"Initializing PaddleOCR (Lang: {self.lang})...")
        try:
            self.ocr_engine = PaddleOCR(
                use_angle_cls=True,
                lang=self.lang,
                enable_mkldnn=False,    # Disable MKL-DNN — not supported on ARM
                cpu_threads=1,          # Single thread to prevent ARM segfault
                show_log=False
            )
            logging.info("PaddleOCR initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize PaddleOCR: {e}")

    def preprocess_image(self, image):
        """
        Preprocesses the image to enhance micro-level fonts.
        NOTE: We do NOT upscale here — the image from the camera is already 1456x1088.
        Upscaling a large image crashes Pi RAM. Instead we sharpen and threshold.
        """
        if image is None:
            return None

        # Resize to a safe resolution for the Pi — large enough for OCR, small enough for RAM
        # 1456x1088 -> 1024x768 (maintains aspect ratio roughly)
        target_w, target_h = 1024, 768
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Apply sharpening kernel to make micro-fonts crisper
        sharpen_kernel = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])
        sharpened = cv2.filter2D(gray, -1, sharpen_kernel)

        # Adaptive threshold — crisp black-on-white text
        processed = cv2.adaptiveThreshold(
            sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Convert back to BGR (3-channel) as PaddleOCR expects color images
        processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

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
            result = self.ocr_engine.ocr(process_img, cls=True)

            extracted_text = []
            if result and result[0]:
                for line in result[0]:
                    try:
                        text = line[1][0]
                        confidence = line[1][1]
                        extracted_text.append(text)
                        logging.info(f"Detected: '{text}' (Confidence: {confidence:.2f})")
                    except (IndexError, TypeError):
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
