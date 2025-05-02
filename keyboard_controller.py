import pyautogui
import time

class KeyboardController:
    """
    Controls keyboard input using PyAutoGUI based on detected ASL signs
    """
    def __init__(self, cooldown=1.0):
        """
        Initialize the keyboard controller
        
        Args:
            cooldown: Minimum time (in seconds) between consecutive key presses
        """
        self.last_key_time = 0
        self.cooldown = cooldown
        self.last_letter = None
        self.consecutive_count = 0
        self.max_consecutive = 3  # Maximum number of consecutive identical predictions before typing
        
    def type_letter(self, letter, confidence):
        """
        Type a letter using PyAutoGUI if it meets criteria
        
        Args:
            letter: The letter to type
            confidence: Confidence of the prediction
            
        Returns:
            typed: Whether a key was typed
        """
        # If no letter was detected or confidence is too low, do nothing
        if letter is None or confidence < 0.7:
            self.consecutive_count = 0
            self.last_letter = None
            return False
            
        current_time = time.time()
        
        # Check if the same letter is being detected consistently
        if letter == self.last_letter:
            self.consecutive_count += 1
        else:
            self.consecutive_count = 1
            self.last_letter = letter
            
        # Only type if:
        # 1. Enough time has passed since the last keypress
        # 2. The same letter has been detected multiple times in a row
        if (current_time - self.last_key_time >= self.cooldown and 
            self.consecutive_count >= self.max_consecutive):
            
            # Type the letter
            pyautogui.write(letter.lower())
            
            # Update the last key time
            self.last_key_time = current_time
            
            # Reset the consecutive counter
            self.consecutive_count = 0
            
            return True
            
        return False
        
    def press_backspace(self):
        """Press the backspace key to delete the last character"""
        current_time = time.time()
        
        if current_time - self.last_key_time >= self.cooldown:
            pyautogui.press('backspace')
            self.last_key_time = current_time
            return True
            
        return False
        
    def press_space(self):
        """Press the space key"""
        current_time = time.time()
        
        if current_time - self.last_key_time >= self.cooldown:
            pyautogui.press('space')
            self.last_key_time = current_time
            return True
            
        return False
        
    def press_enter(self):
        """Press the enter key"""
        current_time = time.time()
        
        if current_time - self.last_key_time >= self.cooldown:
            pyautogui.press('enter')
            self.last_key_time = current_time
            return True
            
        return False