import numpy as np
import math

class ASLRecognizer:
    """
    Recognizes American Sign Language (ASL) hand signs from hand landmarks
    """
    def __init__(self):
        """
        Initialize the ASL recognizer
        """
        # Dictionary to store the last N predictions for smoothing
        self.prediction_history = []
        self.history_length = 5
        self.last_prediction = None
        self.confidence_threshold = 0.7
        
    def _get_angle(self, point1, point2, point3):
        """
        Calculate angle between three points
        
        Args:
            point1, point2, point3: Points as [x, y] coordinates
            
        Returns:
            angle: Angle in degrees
        """
        a = np.array(point1[:2])  # First point (x, y)
        b = np.array(point2[:2])  # Middle point (x, y)
        c = np.array(point3[:2])  # Last point (x, y)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def _get_finger_states(self, landmarks):
        """
        Determine which fingers are extended
        
        Args:
            landmarks: Normalized hand landmarks
            
        Returns:
            finger_states: List of finger states (True if extended, False if folded)
            - [thumb, index, middle, ring, pinky]
        """
        if not landmarks or len(landmarks) < 21:
            return [False, False, False, False, False]
        
        # Fingertip indices
        tip_ids = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
        
        # Base of hand for reference
        wrist = landmarks[0][:2]
        
        # Get the middle of palm as reference point (between landmarks 0 and 9)
        palm_center = [(landmarks[0][0] + landmarks[9][0])/2, 
                       (landmarks[0][1] + landmarks[9][1])/2]
        
        finger_states = []
        
        # Check thumb separately (angle-based)
        thumb_angle = self._get_angle(landmarks[4][:2], landmarks[3][:2], landmarks[2][:2])
        thumb_extended = thumb_angle > 150
        finger_states.append(thumb_extended)
        
        # Check other fingers (height-based)
        for i in range(1, 5):
            # For each finger, check if tip (landmark at tip_ids[i]) is higher than pip
            fingertip = landmarks[tip_ids[i]][:2]
            pip = landmarks[tip_ids[i] - 2][:2]  # PIP joint (two joints below fingertip)
            
            # Calculate distance from tip to wrist and from pip to wrist
            dist_tip_to_palm = math.sqrt((fingertip[0] - palm_center[0])**2 + 
                                         (fingertip[1] - palm_center[1])**2)
            dist_pip_to_palm = math.sqrt((pip[0] - palm_center[0])**2 + 
                                         (pip[1] - palm_center[1])**2)
            
            # Finger is extended if fingertip is farther from palm center than pip
            finger_extended = dist_tip_to_palm > dist_pip_to_palm
            finger_states.append(finger_extended)
            
        return finger_states

    def _get_finger_distances(self, landmarks):
        """
        Calculate distances between fingertips
        
        Args:
            landmarks: Normalized hand landmarks
            
        Returns:
            distances: Dictionary of distances between fingertips
        """
        if not landmarks or len(landmarks) < 21:
            return {}
        
        # Fingertip indices
        tip_ids = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
        
        distances = {}
        
        # Calculate distances between all fingertips
        for i in range(len(tip_ids)):
            for j in range(i+1, len(tip_ids)):
                finger1 = tip_ids[i]
                finger2 = tip_ids[j]
                
                p1 = np.array(landmarks[finger1][:2])
                p2 = np.array(landmarks[finger2][:2])
                
                dist = np.linalg.norm(p1 - p2)
                key = f"{finger1}_{finger2}"
                distances[key] = dist
                
        return distances
    
    def recognize(self, landmarks):
        """
        Classify hand landmarks into an ASL letter
        
        Args:
            landmarks: Normalized hand landmarks from MediaPipe
            
        Returns:
            letter: Recognized ASL letter
            confidence: Confidence score for the prediction
        """
        if not landmarks or len(landmarks) < 21:
            return None, 0
            
        # Get finger states (extended or not)
        finger_states = self._get_finger_states(landmarks)
        
        # Get distances between fingertips
        distances = self._get_finger_distances(landmarks)
        
        # Classify based on finger states and distances
        letter, confidence = self._classify_gesture(finger_states, distances, landmarks)
        
        # Apply smoothing with history
        self.prediction_history.append((letter, confidence))
        if len(self.prediction_history) > self.history_length:
            self.prediction_history.pop(0)
        
        # Get most common prediction in history
        if self.prediction_history:
            predictions = [p[0] for p in self.prediction_history if p[1] >= self.confidence_threshold]
            if predictions:
                from collections import Counter
                letter_counts = Counter(predictions)
                letter = letter_counts.most_common(1)[0][0]
                # Only update if we have a confident prediction
                if letter_counts.most_common(1)[0][1] >= max(2, self.history_length // 2):
                    self.last_prediction = letter
            
        return self.last_prediction, confidence
    
    def _classify_gesture(self, finger_states, distances, landmarks):
        """
        Classify the hand gesture based on finger states and distances
        
        Args:
            finger_states: List of booleans indicating if each finger is extended
            distances: Dictionary of distances between fingertips
            landmarks: Normalized hand landmarks
            
        Returns:
            letter: Recognized ASL letter
            confidence: Confidence score for the prediction
        """
        # Unpack finger states
        thumb_up, index_up, middle_up, ring_up, pinky_up = finger_states
        
        # Default confidence
        confidence = 0.8
        
        # A: Fist with thumb to the side
        if not any([index_up, middle_up, ring_up, pinky_up]) and thumb_up:
            return 'A', confidence
            
        # B: All fingers extended and together, thumb across palm
        elif all([index_up, middle_up, ring_up, pinky_up]) and not thumb_up:
            # Check if fingers are together
            if distances.get('8_12', 999) < 0.3 and distances.get('12_16', 999) < 0.3 and distances.get('16_20', 999) < 0.3:
                return 'B', confidence
                
        # C: Curved hand, fingers together
        elif all([index_up, middle_up, ring_up, pinky_up]) and thumb_up:
            # Check if thumb and index are close
            if distances.get('4_8', 999) < 0.5:
                # Check if all fingers are close together
                if (distances.get('8_12', 999) < 0.3 and 
                    distances.get('12_16', 999) < 0.3 and 
                    distances.get('16_20', 999) < 0.3):
                    return 'C', confidence
                    
        # D: Index finger pointing up, thumb touches middle finger
        elif index_up and not middle_up and not ring_up and not pinky_up:
            if distances.get('4_12', 999) < 0.5:  # Thumb close to middle finger
                return 'D', confidence
                
        # E: All fingers curled, thumb across fingers
        elif not any([index_up, middle_up, ring_up, pinky_up, thumb_up]):
            return 'E', confidence
            
        # F: Index and thumb touching, other fingers extended
        elif middle_up and ring_up and pinky_up and not index_up:
            if distances.get('4_8', 999) < 0.3:  # Thumb and index touching
                return 'F', confidence
                
        # G: Thumb and index extended, index pointing to the side
        elif index_up and thumb_up and not middle_up and not ring_up and not pinky_up:
            # Check index finger orientation
            index_dir = [landmarks[8][0] - landmarks[5][0], landmarks[8][1] - landmarks[5][1]]
            if abs(index_dir[0]) > abs(index_dir[1]):  # More horizontal than vertical
                return 'G', confidence
                
        # H: Index and middle extended together
        elif index_up and middle_up and not ring_up and not pinky_up:
            if distances.get('8_12', 999) < 0.3:  # Index and middle close together
                return 'H', confidence
                
        # I: Pinky extended, others closed
        elif pinky_up and not any([index_up, middle_up, ring_up, thumb_up]):
            return 'I', confidence
            
        # J: Pinky extended, hand moving in J shape (not implemented for static images)
        # For static images, similar to I but with slight rotation
        elif pinky_up and not any([index_up, middle_up, ring_up]):
            # This is a simplification, as J requires motion
            return 'J', 0.5  # Lower confidence due to lack of motion
            
        # K: Index and middle extended, V shape with thumb touching middle finger
        elif index_up and middle_up and not ring_up and not pinky_up:
            if distances.get('8_12', 999) > 0.3:  # Index and middle spread apart
                return 'K', confidence
                
        # L: L shape with thumb and index
        elif index_up and thumb_up and not middle_up and not ring_up and not pinky_up:
            index_dir = [landmarks[8][0] - landmarks[5][0], landmarks[8][1] - landmarks[5][1]]
            thumb_dir = [landmarks[4][0] - landmarks[2][0], landmarks[4][1] - landmarks[2][1]]
            if abs(index_dir[1]) > abs(index_dir[0]) and abs(thumb_dir[0]) > abs(thumb_dir[1]):
                return 'L', confidence
                
        # M: Fingers folded over thumb
        elif not any([index_up, middle_up, ring_up, pinky_up]):
            return 'M', confidence * 0.7  # Lower confidence due to similarity with other letters
            
        # N: Index and middle folded over thumb
        elif not any([index_up, middle_up, ring_up, pinky_up]):
            return 'N', confidence * 0.7  # Lower confidence due to similarity with other letters
            
        # O: Fingers curved to form O shape
        elif not any([index_up, middle_up, ring_up, pinky_up]) and thumb_up:
            if distances.get('4_8', 999) < 0.5:  # Thumb close to index
                return 'O', confidence
                
        # P: Index extended down, thumb to side
        elif not any([index_up, middle_up, ring_up, pinky_up]) and thumb_up:
            return 'P', confidence * 0.7  # Hard to distinguish statically
            
        # Q: Index pointing down, thumb to side
        elif not any([index_up, middle_up, ring_up, pinky_up]) and thumb_up:
            return 'Q', confidence * 0.6  # Very similar to P
            
        # R: Index and middle crossed
        elif index_up and middle_up and not ring_up and not pinky_up:
            # Check if fingers are crossing
            index_tip = np.array(landmarks[8][:2])
            middle_tip = np.array(landmarks[12][:2])
            index_pip = np.array(landmarks[6][:2])
            middle_pip = np.array(landmarks[10][:2])
            
            # Check if lines formed by these points intersect
            def ccw(a, b, c):
                return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
            
            def intersect(a, b, c, d):
                return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)
            
            if intersect(index_pip, index_tip, middle_pip, middle_tip):
                return 'R', confidence
                
        # S: Fist with thumb in front of fingers
        elif not any([index_up, middle_up, ring_up, pinky_up]):
            return 'S', confidence * 0.7  # Similar to other fist-based signs
            
        # T: Fist with thumb between index and middle
        elif not any([index_up, middle_up, ring_up, pinky_up]) and thumb_up:
            return 'T', confidence * 0.7
            
        # U: Index and middle extended together
        elif index_up and middle_up and not ring_up and not pinky_up:
            if distances.get('8_12', 999) < 0.3:  # Index and middle close together
                return 'U', confidence * 0.8  # Similar to H
                
        # V: Index and middle extended in V shape
        elif index_up and middle_up and not ring_up and not pinky_up:
            if distances.get('8_12', 999) > 0.3:  # Index and middle spread apart
                return 'V', confidence
                
        # W: Index, middle, and ring fingers extended
        elif index_up and middle_up and ring_up and not pinky_up:
            return 'W', confidence
            
        # X: Index finger bent at knuckle
        elif not index_up and not middle_up and not ring_up and not pinky_up:
            # Check if index tip is close to index PIP
            index_tip = np.array(landmarks[8][:2])
            index_pip = np.array(landmarks[6][:2])
            if np.linalg.norm(index_tip - index_pip) < 0.2:
                return 'X', confidence
                
        # Y: Thumb and pinky extended
        elif pinky_up and thumb_up and not index_up and not middle_up and not ring_up:
            return 'Y', confidence
            
        # Z: Index finger drawing Z in air (not possible to detect in static image)
        # For static image, approximating with index pointing to the side
        elif index_up and not middle_up and not ring_up and not pinky_up:
            index_dir = [landmarks[8][0] - landmarks[5][0], landmarks[8][1] - landmarks[5][1]]
            if abs(index_dir[0]) > abs(index_dir[1]):  # More horizontal than vertical
                return 'Z', 0.5  # Lower confidence due to lack of motion
                
        # Default case: Unknown gesture
        return None, 0.0