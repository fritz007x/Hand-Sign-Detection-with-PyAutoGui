import cv2
import mediapipe as mp
import numpy as np

class HandDetector:
    """
    Hand tracking and landmark detection using MediaPipe
    """
    def __init__(self, static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize the hand detector with MediaPipe Hands
        
        Args:
            static_image_mode: Whether to treat the input images as a batch or as a video stream
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def find_hands(self, img, draw=True):
        """
        Process an image and detect hands
        
        Args:
            img: Input image (BGR format)
            draw: Whether to draw landmarks on the image
            
        Returns:
            img: Processed image with hand landmarks drawn (if draw=True)
            results: MediaPipe hand detection results
        """
        # Convert BGR image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.hands.process(img_rgb)
        
        # Draw hand landmarks if requested
        if results.multi_hand_landmarks and draw:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return img, results
    
    def find_positions(self, img, results):
        """
        Extract landmark positions from detection results
        
        Args:
            img: Input image
            results: MediaPipe hand detection results
            
        Returns:
            landmark_list: List of normalized landmarks (x, y, z)
            bbox: Bounding box around the hand (x, y, w, h)
        """
        h, w, c = img.shape
        landmark_list = []
        bbox = None
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  # Get the first hand
            
            # Get all landmark positions
            for id, lm in enumerate(hand_landmarks.landmark):
                # Convert normalized coordinates to pixel coordinates
                cx, cy, cz = int(lm.x * w), int(lm.y * h), lm.z
                landmark_list.append([cx, cy, cz])
            
            # Calculate bounding box
            x_coordinates = [landmark[0] for landmark in landmark_list]
            y_coordinates = [landmark[1] for landmark in landmark_list]
            
            x_min, x_max = min(x_coordinates), max(x_coordinates)
            y_min, y_max = min(y_coordinates), max(y_coordinates)
            
            # Add padding to bounding box
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)
            
            bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
            
        return landmark_list, bbox
    
    def get_normalized_landmarks(self, landmark_list):
        """
        Normalize landmarks relative to the wrist position
        
        Args:
            landmark_list: List of landmarks [x, y, z]
            
        Returns:
            normalized_landmarks: Normalized landmarks
        """
        if not landmark_list:
            return []
        
        # Wrist is landmark 0
        wrist_x, wrist_y, wrist_z = landmark_list[0]
        
        # Find the scale using the distance between landmarks 0 (wrist) and 5 (index finger MCP)
        if len(landmark_list) > 5:
            index_mcp_x, index_mcp_y, _ = landmark_list[5]
            scale = ((index_mcp_x - wrist_x) ** 2 + (index_mcp_y - wrist_y) ** 2) ** 0.5
            if scale == 0:  # Avoid division by zero
                scale = 1
        else:
            scale = 1
        
        # Normalize landmarks
        normalized_landmarks = []
        for x, y, z in landmark_list:
            nx = (x - wrist_x) / scale
            ny = (y - wrist_y) / scale
            normalized_landmarks.append([nx, ny, z])
            
        return normalized_landmarks