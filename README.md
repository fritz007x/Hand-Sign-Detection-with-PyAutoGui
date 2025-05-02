# Hand-Sign-Detection-with-PyAutoGui

This is a web app developed for the course CAI2840C-2253-7384 Introduction to Computer Vision. The application uses computer vision to detect and interpret American Sign Language (ASL) hand signs in real-time. When a sign is detected with sufficient confidence, the application can automatically type the corresponding letter using keyboard automation. The app uses PyAutoGUI for writing detected letters. 


## Features


- Real-time hand tracking and landmark detection using MediaPipe
- American Sign Language (ASL) alphabet recognition
- Automatic keyboard typing of detected letters using PyAutoGUI
- Simple OpenCV-based user interface


## System Requirements


- Python 3.10 (MediaPipe is not compatible with Python 3.13)
- Webcam
- Windows operating system


## Components


- `main.py`: The main application with OpenCV interface
- `hand_detector.py`: MediaPipe-based hand tracking implementation
- `asl_recognition.py`: ASL gesture recognition logic
- `keyboard_controller.py`: PyAutoGUI integration for typing



## Installation


1. Make sure you have Python 3.10 installed
2. Clone or download this repository
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```


## Usage


1. Run the application:
   ```
   python main.py
   ```
   or with explicit Python version:
   ```
   C:\path\to\Python310\python.exe main.py
   ```


2. Position your hand clearly in the camera's view
3. Form ASL signs to be recognized
4. Use the following keyboard controls:
   - **T**: Toggle typing mode on/off
   - **C**: Clear the text buffer
   - **Space**: Add a space to the text buffer
   - **Backspace**: Delete the last character
   - **ESC**: Exit the application


## How It Works


### Hand Detection
The application uses MediaPipe's hand tracking model to detect 21 key landmarks on the hand, including fingertips, knuckles, and the wrist.


### ASL Recognition
The detected landmarks are analyzed to:
1. Determine which fingers are extended
2. Calculate distances between key points
3. Identify specific hand configurations corresponding to ASL letters
4. Apply temporal smoothing to prevent flickering between predictions


### Typing Automation
When typing is enabled and a letter is consistently detected:
1. The system waits for multiple consecutive identical detections (default: 3)
2. PyAutoGUI types the recognized letter
3. The typed letter is added to the text buffer displayed on screen


## Tips for Best Performance


- Ensure good, consistent lighting on your hand
- Position your hand at a comfortable distance from the camera
- Make clear, deliberate hand signs
- Hold each sign steady until it's recognized
- Start with simple letters (A, B, C) to get familiar with the system


## Limitations


- Dynamic signs (like J and Z that require motion) are approximated in static form
- Some similar-looking signs (e.g., M and N) may be confused
- Detection accuracy depends on lighting conditions and camera quality
- Designed for right-handed signing (may need adaptation for left-handed users)


## Future Improvements


- Support for full words and phrases
- Machine learning model training for improved accuracy
- Support for dynamic signs and sign language grammar


## License


This project is shared under the MIT License.

