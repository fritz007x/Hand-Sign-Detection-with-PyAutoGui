import cv2
import numpy as np
import time
import pyautogui

# Import custom modules
from hand_detector import HandDetector
from asl_recognition import ASLRecognizer
from keyboard_controller import KeyboardController

def main():
    # Initialize the camera
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
        
    # Set webcam properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Initialize detectors and controllers
    hand_detector = HandDetector(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    asl_recognizer = ASLRecognizer()
    keyboard_controller = KeyboardController(cooldown=1.0)
    
    # Variables for UI
    enable_typing = False
    text_buffer = ""
    last_letter = None
    last_confidence = 0
    
    # Variables for FPS calculation
    fps_time = time.time()
    frame_count = 0
    fps = 0
    
    print("ASL Hand Sign Detection Started")
    print("------------------------------")
    print("Press 'T' to toggle typing")
    print("Press 'C' to clear text buffer")
    print("Press 'SPACE' to add a space")
    print("Press 'BACKSPACE' to delete last character")
    print("Press 'ESC' to exit")
    print("------------------------------")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame.")
            break
            
        # Flip the frame horizontally for a more intuitive mirror view
        frame = cv2.flip(frame, 1)
        
        # Process the frame to detect hands
        img, results = hand_detector.find_hands(frame)
        
        # Get landmark positions
        landmarks, bbox = hand_detector.find_positions(img, results)
        
        # Draw bounding box if hand is detected
        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Normalize landmarks and recognize ASL sign
        if landmarks:
            normalized_landmarks = hand_detector.get_normalized_landmarks(landmarks)
            letter, confidence = asl_recognizer.recognize(normalized_landmarks)
            last_letter = letter
            last_confidence = confidence
            
            # Type the letter if enabled and conditions are met
            if enable_typing and letter:
                if keyboard_controller.type_letter(letter, confidence):
                    text_buffer += letter.lower()
                    
        # Calculate FPS
        frame_count += 1
        current_time = time.time()
        if (current_time - fps_time) > 1:
            fps = frame_count / (current_time - fps_time)
            fps_time = current_time
            frame_count = 0
        
        # Display FPS on frame
        cv2.putText(img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 0, 0), 2, cv2.LINE_AA)
                    
        # Display typing status
        typing_status = "ON" if enable_typing else "OFF"
        cv2.putText(img, f"Typing: {typing_status}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255) if enable_typing else (0, 255, 0), 2, cv2.LINE_AA)
                    
        # Display detected letter
        letter_text = f"Letter: {last_letter if last_letter else 'None'}"
        cv2.putText(img, letter_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 0, 255), 2, cv2.LINE_AA)
                    
        # Display confidence bar if a letter is detected
        if last_letter:
            # Confidence bar background
            cv2.rectangle(img, (10, 130), (210, 150), (0, 0, 0), cv2.FILLED)
            # Confidence bar fill
            conf_width = int(last_confidence * 200)
            cv2.rectangle(img, (10, 130), (10 + conf_width, 150), (0, 255, 0), cv2.FILLED)
            # Confidence text
            cv2.putText(img, f"Confidence: {int(last_confidence * 100)}%", (10, 145), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                        
        # Display text buffer
        # Background for text display
        cv2.rectangle(img, (0, img.shape[0] - 40), (img.shape[1], img.shape[0]), (0, 0, 0), cv2.FILLED)
        cv2.putText(img, f"Text: {text_buffer[-40:] if len(text_buffer) > 40 else text_buffer}", 
                    (10, img.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Display instructions
        cv2.putText(img, "T: Toggle typing | C: Clear | ESC: Exit", 
                    (10, img.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        
        # Display the resulting frame
        cv2.imshow('ASL Hand Sign Detection', img)
        
        # Process key presses
        key = cv2.waitKey(1) & 0xFF
        
        # ESC key - exit
        if key == 27:
            break
            
        # T key - toggle typing
        elif key == ord('t') or key == ord('T'):
            enable_typing = not enable_typing
            print(f"Typing {'enabled' if enable_typing else 'disabled'}")
            
        # C key - clear text buffer
        elif key == ord('c') or key == ord('C'):
            text_buffer = ""
            print("Text buffer cleared")
            
        # Space key - add space
        elif key == 32:  # Space key
            if enable_typing:
                if keyboard_controller.press_space():
                    text_buffer += " "
                    
        # Backspace key - delete last character
        elif key == 8:  # Backspace key
            if enable_typing and text_buffer:
                if keyboard_controller.press_backspace():
                    text_buffer = text_buffer[:-1]
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed")

if __name__ == "__main__":
    main()
