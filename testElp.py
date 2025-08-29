import cv2
import time
import numpy as np

# --- Configuration for ELP Camera and Detection ---
# IMPORTANT: You may need to change these values based on your camera model and system.
# Use a high-quality ELP camera that supports a global shutter for best results with fast-moving insects.

# Camera device index. This might be different on your system (e.g., 0, 1, 2, ...).
# You can find the correct index by trying different numbers or using a tool like `v4l2-ctl --list-devices` on Linux.
camera_index = 1

# Set desired resolution (width, height)
# Higher resolution captures more detail, but might lower the frame rate.
desired_width = 1920
desired_height = 1080

# Set desired frame rate (FPS)
# Higher FPS is crucial for capturing fast-moving insects without missing them.
desired_fps =30

# --- Function to configure and test the camera ---
def configure_camera(cap):
    """
    Configures the camera with the desired resolution and frame rate.
    Attempts to enable global shutter, though this is camera-dependent.
    """
    print(f"Attempting to set resolution to {desired_width}x{desired_height}...")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
    
    print(f"Attempting to set frame rate to {desired_fps} FPS...")
    cap.set(cv2.CAP_PROP_FPS, desired_fps)
    
    # Attempt to enable global shutter.
    # The property ID might vary, or the camera might not support it.
    # Some cameras might have a specific vendor-defined property for global shutter mode.
    # The default value for CAP_PROP_TRIGGER is usually 0. You might need to check your camera's manual.
    try:
        cap.set(cv2.CAP_PROP_TRIGGER, 1) # This is a generic attempt; specific cameras may have different properties.
        print("Attempted to enable global shutter. Please check your camera manual for specific settings.")
    except Exception as e:
        print(f"Could not set global shutter property: {e}")

    # Verify the settings
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print("\n--- Camera Properties ---")
    print(f"Actual Resolution: {actual_width}x{actual_height}")
    print(f"Actual Frame Rate: {actual_fps} FPS")
    
    if actual_width != desired_width or actual_height != desired_height:
        print("Warning: Resolution could not be set as requested.")
    if actual_fps < desired_fps:
        print("Warning: Frame rate is lower than requested. Check your camera's max FPS.")

    return actual_width, actual_height

# --- Simple insect detection placeholder ---
def detect_insects(frame):
    """
    A placeholder function for insect detection.
    This simple example uses motion detection.
    A real-world application would use more advanced techniques (e.g., background subtraction, YOLO).
    """
    # Convert frame to grayscale for motion detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Use a simple threshold to find moving objects
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    insect_count = 0
    for contour in contours:
        # Filter out small noise
        if cv2.contourArea(contour) > 500:
            insect_count += 1
            x, y, w, h = cv2.boundingRect(contour)
            # Draw a green rectangle around the detected "insect"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return frame, insect_count

# --- Main execution loop ---
def main():
    # Open the video capture device
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not open camera with index {camera_index}.")
        return

    # Configure the camera
    width, height = configure_camera(cap)

    # Main video processing loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        # Process the frame to detect insects
        processed_frame, insect_count = detect_insects(frame)

        # Display the resulting frame
        cv2.imshow('Insect Detection', processed_frame)
        print(f"Insects detected in frame: {insect_count}", end='\r')

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
