import sys
import time
import cv2
import numpy as np

# Add the ECG directory to the python path so we can import camera_core
# Adjust this path if the script is moved to a different relative location
sys.path.append('d:/ECG')

try:
    from camera_core import CameraManager
except ImportError:
    print("Error: Could not import CameraManager from d:/ECG/camera_core.py")
    sys.exit(1)

from ocr_worker import OCRWorker

def main():
    print("Initializing OCR Engine...")
    # Initialize OCR. Set use_gpu=False for Raspberry Pi CPU.
    ocr = OCRWorker(use_gpu=False, lang='en')
    
    if ocr.ocr_engine is None:
        print("Failed to load OCR engine. Please check dependencies.")
        return

    print("\nInitializing Camera...")
    cam_manager = CameraManager()
    
    try:
        # We start the camera but don't necessarily start recording video.
        # The initialized camera will let us capture frames.
        cam_manager.initialize_camera()
        time.sleep(2) # Give the camera sensor a moment to warm up/adjust exposure
        print("Camera ready.")
        
        while True:
            # Trigger Mechanism: Wait for user input
            user_input = input("\nPress 'Enter' to capture and read text, or 'q' to quit: ")
            
            if user_input.lower() == 'q':
                print("Exiting...")
                break
                
            print("Capturing frame...")
            # We capture a frame. 
            # Note: get_frame_jpeg returns a JPEG byte array.
            # We need to decode it back to a numpy array for OpenCV/PaddleOCR.
            jpeg_bytes = cam_manager.get_frame_jpeg()
            
            if jpeg_bytes is None:
                print("Failed to capture frame from camera.")
                continue
                
            # Decode JPEG bytes to numpy array
            nparr = np.frombuffer(jpeg_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                print("Failed to decode the captured frame.")
                continue
                
            print(f"Frame captured successfully. Size: {frame.shape}")
            
            # Extract text using our worker
            # The worker automatically preprocesses (upscales & thresholds) the image
            print("Processing image for micro-level text...")
            ocr.extract_text(frame, preprocess=True)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Cleaning up camera resources...")
        cam_manager.close()

if __name__ == "__main__":
    main()
