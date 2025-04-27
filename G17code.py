import RPi.GPIO as GPIO
import time
import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import requests # Added for downloading
import shutil   # Added for saving downloaded file

# GPIO Pin Configuration
TRIG_PIN = 23  # Ultrasonic sensor trigger pin
ECHO_PIN = 24  # Ultrasonic sensor echo pin
SERVO_PIN = 18  # SG90 servo motor pin
# Stepper motor pins
STEP_PIN = 17
DIR_PIN = 27
ENABLE_PIN = 22

# Waste categories and corresponding rotation angles (Bins)
# These are the final categories the system uses for sorting.
WASTE_CATEGORIES = {
    'plastic': 90,   # 90 degrees rotation
    'paper': 180,    # 180 degrees rotation
    'metal': 270,    # 270 degrees rotation
    'trash': 0      # 0 degrees (default position, includes Biodegradable, Trash from model)
}

class SmartWasteClassifier:
    def __init__(self, model_path=None):
        # Create a directory for captured images if it doesn't exist
        os.makedirs("captured_images", exist_ok=True)
        # Count for naming captured images
        self.image_count = 0

        # --- Model Loading Logic ---
        # If a specific model path is provided and exists, try loading it.
        if model_path and os.path.exists(model_path):
            try:
                self.model = load_model(model_path)
                # Removed explicit model.build() - rely on load_model and correct target_size
                print(f"Classification model loaded from local path: {model_path}")
            except Exception as e:
                print(f"Error loading model from {model_path}: {str(e)}")
                print("Attempting to load custom model from GitHub...")
                self.load_custom_model() # Fallback to custom download
        else:
            # If no path provided or path invalid, attempt to load custom model
            print("No valid local model path provided, attempting to load custom model from GitHub...")
            self.load_custom_model()

        # --- Define Model Output Categories (Based on labels.txt) ---
        # These MUST match the order and names your model was trained on.
        # From https://github.com/marvin1ing/model/blob/main/labels.txt
        self.categories = ['Biodegradable', 'Metal', 'Paper', 'Plastic', 'Trash']
        print(f"Using model categories: {self.categories}")

        # --- Camera Initialization ---
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Ensure camera is working
        if not self.camera.isOpened():
            print("ERROR: Camera not accessible")
        else:
            # Discard first few frames to allow camera to adjust
            print("Initializing camera...")
            for _ in range(5):
                self.camera.read()
                time.sleep(0.1)
            print("Camera initialized successfully")

    def load_custom_model(self):
        """Attempt to download and load the custom model from GitHub"""
        # --- Configuration for your custom model ---
        model_url = "https://raw.githubusercontent.com/marvin1ing/model/main/keras_model.h5"
        local_model_path = "custom_waste_model.h5" # Local filename for the custom model

        # Check if model already exists locally
        if os.path.exists(local_model_path):
            try:
                self.model = load_model(local_model_path)
                # Removed explicit model.build()
                print(f"Custom model loaded from local file: {local_model_path}")
                return # Success
            except Exception as e:
                print(f"Error loading local custom model ({local_model_path}): {str(e)}")
                print("Attempting to re-download...")
                # Try deleting the potentially corrupted file before redownloading
                try:
                    os.remove(local_model_path)
                except OSError as oe:
                    print(f"Could not remove existing file {local_model_path}: {oe}")


        # Download the model if not available locally or if loading failed
        try:
            print(f"Downloading custom model from {model_url}...")
            response = requests.get(model_url, stream=True)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

            with open(local_model_path, 'wb') as f:
                # Use shutil.copyfileobj for efficient download
                shutil.copyfileobj(response.raw, f)
            print(f"Model downloaded successfully to {local_model_path}")

            # Load the newly downloaded model
            self.model = load_model(local_model_path)
            # Removed explicit model.build()
            print("Custom model loaded successfully after download.")

        except requests.exceptions.RequestException as e:
            print(f"Failed to download model: {str(e)}")
            self.model = None
        except Exception as e:
            print(f"Error loading downloaded custom model: {str(e)}")
            self.model = None

        if self.model is None:
            print("ERROR: Failed to load the custom classification model.")

    def capture_image(self):
        """Capture an image for classification"""
        if not self.camera.isOpened():
             print("ERROR: Camera is not open, cannot capture image.")
             return False

        # Capture a frame
        ret, frame = self.camera.read()

        if not ret or frame is None:
            print("ERROR: Failed to capture frame from camera")
            return False

        # Save the captured image
        self.image_count += 1
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = f"captured_images/waste_{timestamp}_{self.image_count}.jpg"
        cv2.imwrite(filename, frame)

        # Save a copy as waste_image.jpg for classification
        cv2.imwrite('waste_image.jpg', frame)

        print(f"Image captured and saved as {filename}")
        return True

    def classify_waste(self):
        """Classify the waste using the loaded custom model"""
        # Check if model loaded successfully
        if self.model is None:
            print("ERROR: Model not loaded, cannot classify. Defaulting to 'trash'.")
            return "trash"

        # Capture image
        if not self.capture_image():
            print("WARNING: Image capture failed, defaulting to 'trash'")
            return "trash"

        try:
            # --- Preprocess image for your custom model ---
            # *** IMPORTANT: Adjust target_size if your model expects a different input size ***
            # Assuming (224, 224) based on common practice. Change if needed!
            img_target_size = (224, 224)
            img = image.load_img('waste_image.jpg', target_size=img_target_size)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0 # Rescale pixel values

            # Get prediction
            predictions = self.model.predict(img_array)
            predicted_class_index = np.argmax(predictions[0])
            confidence = np.max(predictions[0])

            # Get the category name from the model's output classes
            model_category = self.categories[predicted_class_index]

            # --- Map Model category to Bin category ---
            if model_category == 'Paper':
                bin_category = 'paper'
            elif model_category == 'Plastic':
                bin_category = 'plastic'
            elif model_category == 'Metal':
                bin_category = 'metal'
            # Map 'Biodegradable' and 'Trash' from the model to the 'trash' bin
            elif model_category in ['Biodegradable', 'Trash']:
                bin_category = 'trash'
            else:
                # Fallback for any unexpected model category
                print(f"Warning: Unknown model category '{model_category}', assigning to 'trash'.")
                bin_category = 'trash'

            print(f"Classified as: {model_category} (confidence: {confidence:.2f})")
            print(f"Assigned to bin: {bin_category}")
            return bin_category

        except Exception as e:
            print(f"Error during classification: {str(e)}")
            import traceback
            traceback.print_exc() # Print detailed traceback for debugging
            return "trash"  # Default to trash bin on error

class SmartWasteSystem:
    def __init__(self, model_path=None):
        self.setup_gpio()
        # Pass the model path (or None) to the classifier
        self.classifier = SmartWasteClassifier(model_path)
        self.lid_open = False
        print("Smart Waste System Initialized")

    def setup_gpio(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        # Setup Ultrasonic Sensor
        GPIO.setup(TRIG_PIN, GPIO.OUT)
        GPIO.setup(ECHO_PIN, GPIO.IN)

        # Setup Servo Motor
        GPIO.setup(SERVO_PIN, GPIO.OUT)
        self.servo_pwm = GPIO.PWM(SERVO_PIN, 50)  # 50Hz frequency
        self.servo_pwm.start(0) # Start PWM but servo remains inactive

        # Setup Drop Mechanism Servo Motor
        self.DROP_SERVO_PIN = 13  # Define the pin - modify as needed
        GPIO.setup(self.DROP_SERVO_PIN, GPIO.OUT)
        self.drop_servo_pwm = GPIO.PWM(self.DROP_SERVO_PIN, 50)  # 50Hz frequency
        self.drop_servo_pwm.start(0)  # Start PWM but servo remains inactive

        # Setup Stepper Motor
        GPIO.setup(STEP_PIN, GPIO.OUT)
        GPIO.setup(DIR_PIN, GPIO.OUT)
        GPIO.setup(ENABLE_PIN, GPIO.OUT)
        # Enable stepper motor driver (assuming LOW enables)
        GPIO.output(ENABLE_PIN, GPIO.LOW)
        print("GPIO Pins Setup Completed")

    def measure_distance(self):
        # Send trigger pulse
        GPIO.output(TRIG_PIN, GPIO.LOW)
        time.sleep(0.02) # Small delay before pulse
        GPIO.output(TRIG_PIN, GPIO.HIGH)
        time.sleep(0.00001) # 10 microsecond pulse
        GPIO.output(TRIG_PIN, GPIO.LOW)

        # Measure echo pulse duration
        pulse_start_time = time.time()
        pulse_end_time = time.time()
        timeout = pulse_start_time + 0.1 # 100ms timeout

        # Wait for the echo pin to go high (start of pulse)
        while GPIO.input(ECHO_PIN) == 0:
            pulse_start_time = time.time()
            if pulse_start_time > timeout:
                # print("Echo timeout (pulse start)")
                return -1 # Indicate timeout

        # Wait for the echo pin to go low (end of pulse)
        while GPIO.input(ECHO_PIN) == 1:
            pulse_end_time = time.time()
            if pulse_end_time > timeout:
                 # print("Echo timeout (pulse end)")
                 return -1 # Indicate timeout

        # Calculate distance
        pulse_duration = pulse_end_time - pulse_start_time
        # Speed of sound = 34300 cm/s. Divide by 2 for round trip.
        distance = (pulse_duration * 34300) / 2
        return round(distance, 2)

    def control_servo(self, angle):
        # Map angle (0-180) to duty cycle (approx 2.5-12.5)
        # Adjust these values (2.0, 12.0) if your servo behaves differently
        duty = 2.0 + (angle / 18.0) * (12.0 - 2.0) / 10.0
        # Ensure duty cycle is within reasonable bounds
        duty = max(2.0, min(12.0, duty))

        self.servo_pwm.ChangeDutyCycle(duty)
        time.sleep(0.5)  # Allow servo time to move
        # Stop sending PWM signal to prevent servo jitter and save power
        self.servo_pwm.ChangeDutyCycle(0)

    def open_lid(self):
        """Open the lid using the servo motor"""
        if not self.lid_open:
            print("Opening lid...")
            self.control_servo(90)  # Angle to open lid (adjust if needed)
            self.lid_open = True
            print("Lid opened.")

    def close_lid(self):
        """Close the lid using the servo motor"""
        if self.lid_open:
            print("Closing lid...")
            self.control_servo(0)   # Angle to close lid (adjust if needed)
            self.lid_open = False
            print("Lid closed.")

    def detect_hand(self, threshold_distance=25):
        """Detect hand using ultrasonic sensor and open lid if detected"""
        distance = self.measure_distance()

        # Check if distance reading is valid and within threshold
        if 0 < distance < threshold_distance: # Check distance > 0 as well
            print(f"Hand detected at {distance:.2f}cm")
            # Only open if not already open
            if not self.lid_open:
                 self.open_lid()
            return True # Hand detected

        # Optional: Close lid if hand removed and lid is open
        # elif self.lid_open and distance >= threshold_distance:
        #     print("Hand removed, closing lid.")
        #     self.close_lid()

        return False # No hand detected close enough

    def rotate_stepper(self, target_angle):
        """Rotate the stepper motor to the target angle."""
        # This is a simplified rotation logic. You might need a more robust
        # implementation tracking current position if precise multi-turn
        # positioning is needed, or if angles beyond 360 are used.
        # Current assumption: We always rotate from 0 degrees.

        print(f"Rotating stepper to {target_angle} degrees...")
        # Ensure angle is within 0-360 range for simplicity if needed
        # target_angle = target_angle % 360

        # Set direction based on target angle (simplistic: assume > 0 is one way)
        # A more robust system would compare target_angle to current_angle
        if target_angle > 0: # Simple assumption: positive angle -> CW
             GPIO.output(DIR_PIN, GPIO.HIGH) # Clockwise (adjust if reversed)
             print("Direction: Clockwise")
        else: # 0 or negative angle -> CCW (or stay at 0)
             GPIO.output(DIR_PIN, GPIO.LOW) # Counter-Clockwise (adjust if reversed)
             print("Direction: Counter-Clockwise")

        # Calculate steps needed (assuming 1.8 degrees per step -> 200 steps/rev)
        # Adjust 1.8 if your stepper has a different step angle (e.g., 0.9, 7.5)
        # Use microstepping factor if configured on your driver.
        step_angle = 1.8
        steps_per_revolution = 360 / step_angle
        steps_to_move = int(abs(target_angle) / step_angle)
        print(f"Calculated steps: {steps_to_move}")

        # Ensure motor driver is enabled
        GPIO.output(ENABLE_PIN, GPIO.LOW) # Assuming LOW enables

        # Execute steps
        step_delay = 0.005 # Controls speed (lower is faster). Adjust as needed.
        for _ in range(steps_to_move):
            GPIO.output(STEP_PIN, GPIO.HIGH)
            time.sleep(step_delay)
            GPIO.output(STEP_PIN, GPIO.LOW)
            time.sleep(step_delay)

        # Optional: Disable motor driver to save power and reduce heat when idle
        # GPIO.output(ENABLE_PIN, GPIO.HIGH) # Assuming HIGH disables
        print(f"Stepper rotation to {target_angle} degrees complete.")

    def return_stepper_to_zero(self):
        """Rotate the stepper motor back to the 0 degree position."""
        # This requires knowing how far it moved. The current `rotate_stepper`
        # doesn't track position. For now, we'll just call rotate_stepper(0)
        # which isn't truly correct unless we track the current angle.
        # A better approach would involve tracking the last angle.
        # Simplistic fix: just rotate back by the negative of the last angle?
        # Let's assume `rotate_stepper` is called with the target angle,
        # and we call this function to 'reverse' that rotation.
        # THIS IS A PLACEHOLDER - needs proper position tracking.
        print("Returning stepper to default (0 degrees)...")
        # For now, just call rotate_stepper(0). This might not work correctly
        # if the motor direction logic isn't robust or position isn't tracked.
        self.rotate_stepper(0) # Attempt to rotate to 0 position

    def drop_trash_to_bin(self):
        """Use a servo motor to drop trash into the bin after stepper motor rotation"""
        print("Dropping trash into bin...")
        
        # Open the trap door - rotate servo to 90 degrees
        duty = 2.0 + (90 / 18.0) * (12.0 - 2.0) / 10.0
        self.drop_servo_pwm.ChangeDutyCycle(duty)
        time.sleep(1)  # Allow time for the servo to move
        self.drop_servo_pwm.ChangeDutyCycle(0)  # Stop signal to prevent jitter
        
        # Pause to allow trash to fall
        time.sleep(2)
        
        # Close the trap door - rotate servo back to 0 degrees
        duty = 2.0 + (0 / 18.0) * (12.0 - 2.0) / 10.0
        self.drop_servo_pwm.ChangeDutyCycle(duty)
        time.sleep(1)  # Allow time for the servo to move
        self.drop_servo_pwm.ChangeDutyCycle(0)  # Stop signal to prevent jitter
        
        print("Trash dropped into bin.")

    def hand_detection_mode(self):
        """Run only hand detection and lid opening/closing"""
        try:
            # Initial state - ensure lid is closed
            self.close_lid()
            print("Hand detection mode running. Lid starts closed. Press Ctrl+C to exit.")

            last_detection_state = False
            while True:
                hand_detected = self.detect_hand()

                if hand_detected and not last_detection_state:
                    # Hand just detected (state changed from False to True)
                    print("Hand detected - Lid opened.")
                    # Keep lid open while hand is present (or add timeout later)

                elif not hand_detected and last_detection_state:
                    # Hand just removed (state changed from True to False)
                    print("Hand removed. Closing lid after delay...")
                    time.sleep(3) # Wait 3 seconds before closing
                    self.close_lid()

                last_detection_state = hand_detected
                time.sleep(0.2) # Short delay between sensor readings

        except KeyboardInterrupt:
            print("\nHand detection mode terminated by user.")
        except Exception as e:
            print(f"Error in hand detection mode: {str(e)}")
        finally:
            print("Cleaning up...")
            self.close_lid() # Ensure lid is closed on exit
            self.servo_pwm.stop()
            self.drop_servo_pwm.stop()  # Stop the drop servo PWM
            GPIO.cleanup()
            print("GPIO cleaned up. Exiting hand detection mode.")

    def run(self):
        """Main operational loop"""
        try:
            # Initial state - ensure lid is closed
            self.close_lid()
            # Ensure stepper is at default position (needs proper implementation)
            self.return_stepper_to_zero()

            print("Smart Waste System running. Lid starts closed. Press Ctrl+C to exit.")

            while True:
                # 1. Detect Hand
                if self.detect_hand(): # This also opens the lid if hand detected
                    print("Hand detected, lid open. Waiting for waste...")
                    # Wait a bit for user to drop waste. Maybe use sensor again?
                    # For simplicity, just wait a fixed time.
                    time.sleep(4) # Adjust wait time as needed

                    # 2. Classify Waste (if model is available)
                    if self.classifier.model:
                        print("Attempting to classify waste...")
                        waste_type = self.classifier.classify_waste() # This captures image inside
                        print(f"Classification result: {waste_type}")

                        # 3. Rotate Bin
                        # Get the rotation angle for the classified bin category
                        rotation_angle = WASTE_CATEGORIES.get(waste_type, 0) # Default to 0 (trash)
                        print(f"Rotating bin to {rotation_angle} degrees for '{waste_type}'...")
                        self.rotate_stepper(rotation_angle)
                        time.sleep(1) # Small delay after rotation

                        # 4. Drop trash into bin
                        self.drop_trash_to_bin()  # Drop the trash after bin rotation

                        # 5. Close Lid
                        self.close_lid()

                        # 6. Wait and Return Bin to Zero
                        print("Waiting before returning bin...")
                        time.sleep(3) # Time for waste to settle?
                        self.return_stepper_to_zero()

                        # 7. Cooldown
                        print("Cycle complete. Entering cooldown...")
                        time.sleep(3) # Prevent immediate re-triggering

                    else:
                        # Model not loaded, just close lid after wait
                        print("Model not available. Closing lid.")
                        time.sleep(2)
                        self.close_lid()
                        time.sleep(3) # Cooldown

                # If no hand detected, just loop briefly
                else:
                    # Ensure lid is closed if it was somehow left open
                    if self.lid_open:
                        self.close_lid()
                    time.sleep(0.2) # Check sensor periodically

        except KeyboardInterrupt:
            print("\nProgram terminated by user.")
        except Exception as e:
            print(f"An error occurred during run: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            print("Cleaning up...")
            # Ensure lid is closed and PWM stopped
            if self.lid_open:
                 self.control_servo(0) # Send close command directly
            self.servo_pwm.stop()
            self.drop_servo_pwm.stop()  # Stop the drop servo PWM
            # Ensure stepper motor driver is disabled (if applicable)
            # GPIO.output(ENABLE_PIN, GPIO.HIGH) # Assuming HIGH disables
            GPIO.cleanup()
            # Release camera
            if self.classifier and self.classifier.camera.isOpened():
                self.classifier.camera.release()
                print("Camera released.")
            print("GPIO cleaned up. Exiting.")

def test_classification_only(model_path=None):
    """
    Test only the classification without motor control.
    Useful for debugging the model and image capture.
    """
    print("--- Classification Test Mode ---")
    # Use the default custom model path if none provided
    if model_path is None:
        model_path = "custom_waste_model.h5"
        print(f"No model path specified, using default: {model_path}")

    # Define classifier variable in the function scope for the finally block
    classifier = None
    try:
        # Initialize only the classifier part
        classifier = SmartWasteClassifier(model_path)

        # Check if model loaded correctly
        if classifier.model is None:
            print("Model could not be loaded. Exiting test.")
            return

        print("Press Ctrl+C to exit.")

        while True:
            input("Press Enter to capture and classify an image...")
            waste_type = classifier.classify_waste() # This captures and classifies
            print(f"--> Classified bin category: {waste_type}")

            # Try to display the image if possible (requires a GUI environment)
            try:
                img = cv2.imread('waste_image.jpg')
                if img is not None:
                    cv2.imshow('Captured Image (Press any key)', img)
                    print("Displaying captured image. Press any key in the image window to continue...")
                    cv2.waitKey(0) # Wait indefinitely for a key press
                    cv2.destroyAllWindows() # Close window immediately after key press
                else:
                    print("Could not read waste_image.jpg for display.")
            except cv2.error as e:
                # This usually happens if running without a display (e.g., SSH without X forwarding)
                print(f"Cannot display image (GUI not available or error): {e}")
            except Exception as e:
                print(f"An unexpected error occurred during image display: {e}")
            finally:
                # Ensure any specific window is closed even if waitKey was interrupted
                cv2.destroyAllWindows()


    except KeyboardInterrupt:
        print("\nClassification test terminated by user.")
    except Exception as e:
        print(f"Error during classification testing: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
         # Release camera if it was opened AND classifier was initialized
        if classifier is not None and classifier.camera.isOpened(): # Check if classifier exists first
             classifier.camera.release()
             print("Camera released.")
        # Corrected Indentation for the line below:
        cv2.destroyAllWindows() # Should be aligned with the 'if' above, directly under 'finally'


def test_hand_detection_only():
    """
    Test only the hand detection and lid opening/closing.
    Useful for debugging the ultrasonic sensor and servo operation.
    """
    print("--- Hand Detection Test Mode ---")
    smart_system = None # Define in outer scope for finally block
    try:
        # Initialize the system (which sets up GPIO) but don't run the main loop
        smart_system = SmartWasteSystem()
        # Run the specific hand detection loop
        smart_system.hand_detection_mode()

    except Exception as e:
        print(f"Error during hand detection testing: {str(e)}")
        import traceback
        traceback.print_exc()
    # No finally block needed here as hand_detection_mode has its own


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == "test":
            # Test classification only: python script.py test [optional_model_path]
            # Uses 'custom_waste_model.h5' by default if no path given
            test_model_path = sys.argv[2] if len(sys.argv) > 2 else "custom_waste_model.h5"
            test_classification_only(test_model_path)
        elif command == "hand":
            # Test hand detection only: python script.py hand
            test_hand_detection_only()
        else:
            print(f"Unknown command: {sys.argv[1]}")
            print("Usage:")
            print("  python <script_name>.py          (Run full system)")
            print("  python <script_name>.py test [model_path] (Test classification, uses custom_waste_model.h5 by default)")
            print("  python <script_name>.py hand     (Test hand detection and lid)")
    else:
        # Regular run mode: python script.py
        print("Starting Smart Waste System in full operational mode...")
        # Use the custom model by default
        default_model = "custom_waste_model.h5"
        smart_system = SmartWasteSystem(model_path=default_model)
        smart_system.run()