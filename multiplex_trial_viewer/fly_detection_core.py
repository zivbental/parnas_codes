"""
Fly detection core module for multiplex trial viewer.
Extracted from experiment_dashboard.py to provide standalone fly detection functionality.
"""

import cv2
import numpy as np


class FlyDetector:
    """
    Fly detection class that maintains background model and processes frames.
    """
    
    def __init__(self, alpha=0.15):
        """
        Initialize the fly detector.
        
        Args:
            alpha (float): Weight for the running average background model (default: 0.15)
        """
        self.background_model = None
        self.alpha = alpha
    
    def update_background_model(self, frame):
        """
        Updates the background model using a running average.
        
        Args:
            frame (numpy.ndarray): Input frame (grayscale)
            
        Returns:
            numpy.ndarray: Updated background model
        """
        # Initialize the background model if it's the first frame
        if self.background_model is None:
            self.background_model = frame.copy().astype("float")
            return self.background_model

        # Update the background model with the current frame
        cv2.accumulateWeighted(frame, self.background_model, self.alpha)

        # Return the updated background model
        return self.background_model

    def fly_detection(self, frame, mask):
        """
        This function processes the given frame for fly detection by identifying significant changes in the image
        within the specified mask area, and returns the processed frame with detected flies highlighted.

        Args:
            frame (numpy.ndarray): The current video frame.
            mask (numpy.ndarray): The mask specifying the area to focus on.

        Returns:
            tuple: (processed_frame, valid_contours) - The processed frame and list of valid contours
        """
        # Convert frame to grayscale (assuming background subtraction works best in single channel)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to the grayscale frame to reduce noise
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        
        # Apply the mask to the blurred frame
        masked_frame = cv2.bitwise_and(blurred_frame, blurred_frame, mask=mask)
        
        # Update the background model with the masked frame
        background_model = self.update_background_model(masked_frame)
        
        # Convert the background model to uint8 before subtraction
        background_model_uint8 = cv2.convertScaleAbs(background_model)
        
        # Subtract the background from the masked frame
        diff_frame = cv2.absdiff(background_model_uint8, masked_frame)
        
        # Apply a threshold to get a binary image of motion areas
        _, thresh_frame = cv2.threshold(diff_frame, 5, 255, cv2.THRESH_BINARY)  # Adjust threshold value as needed
        
        # Apply morphological operations to remove noise and fill gaps
        kernel = np.ones((3, 3), np.uint8)
        processed_frame = cv2.morphologyEx(thresh_frame, cv2.MORPH_CLOSE, kernel)
        processed_frame = cv2.morphologyEx(processed_frame, cv2.MORPH_OPEN, kernel)
        
        # Find contours (which will be potential flies)
        contours, _ = cv2.findContours(processed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Initialize a list to hold valid contours
        valid_contours = []

        # Draw bounding boxes around detected flies and filter contours
        for contour in contours:
            if 1 < cv2.contourArea(contour) < 200:  # Filter out small contours/noise, adjust value as needed
                x, y, w, h = cv2.boundingRect(contour)
                # Draw rectangle on the frame: (frame, top-left corner, bottom-right corner, color, thickness)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Add the valid contour to the list
                valid_contours.append(contour)

        # Return the processed frame and only the valid contours
        return processed_frame, valid_contours

    def assign_contours_to_chambers(self, contours, chambers):
        """
        Assigns contours (flies) to their respective chambers and calculates the fly's location within each chamber's x-axis.

        Args:
            contours (list): A list of detected contours (flies).
            chambers (dict): A dictionary containing the chamber configuration, with each chamber's x, y, width, and height.

        Returns:
            dict: A dictionary where each key is a chamber ID and the value is the normalized x-axis location of the fly within that chamber.
                  If no fly is detected in a chamber, the value is NaN.
        """
        chamber_fly_locations = {}

        for chamber_id, config in chambers.items():
            x, y, width, height = config['x'], config['y'], config['width'], config['height']
            chamber_center_x = x + width / 2
            
            fly_x_positions = []

            for contour in contours:
                # Calculate the center of the contour
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    contour_center_x = int(M['m10'] / M['m00'])
                    contour_center_y = int(M['m01'] / M['m00'])

                    # Check if the contour is within the current chamber's x and y bounds
                    if x <= contour_center_x <= x + width and y <= contour_center_y <= y + height:
                        # Normalize the x position within the chamber
                        normalized_x = ((contour_center_x - x) / width) * 200 - 100
                        fly_x_positions.append(normalized_x)
            
            # If one or more flies are detected in the chamber, return the average position
            if fly_x_positions:
                chamber_fly_locations[chamber_id] = np.mean(fly_x_positions)

        return chamber_fly_locations

    def process_frame_with_chambers(self, frame, chambers):
        """
        Process a single frame with chamber detection.
        
        Args:
            frame (numpy.ndarray): Input video frame
            chambers (dict): Chamber configuration dictionary
            
        Returns:
            tuple: (processed_frame, fly_locations) - Processed frame and detected fly locations
        """
        # Create mask for all chambers
        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        
        for chamber_id, config in chambers.items():
            x, y, width, height = config['x'], config['y'], config['width'], config['height']
            cv2.rectangle(mask, (x, y), (x + width, y + height), 255, -1)
        
        # Apply fly detection
        processed_frame, valid_contours = self.fly_detection(frame, mask)
        
        # Assign contours to chambers
        fly_locations = self.assign_contours_to_chambers(valid_contours, chambers)
        
        return processed_frame, fly_locations


def test_fly_detection():
    """
    Test function for fly detection core functionality.
    """
    print("Testing fly detection core...")
    
    # Create a test detector
    detector = FlyDetector()
    
    # Create a test frame (black image with white rectangle)
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_frame, (100, 100), (200, 120), (255, 255, 255), -1)
    
    # Create test chambers configuration
    test_chambers = {
        'chamber_1': {'x': 100, 'y': 100, 'width': 100, 'height': 20}
    }
    
    # Process the frame
    processed_frame, fly_locations = detector.process_frame_with_chambers(test_frame, test_chambers)
    
    print(f"Processed frame shape: {processed_frame.shape}")
    print(f"Detected fly locations: {fly_locations}")
    print("Fly detection core test completed successfully!")
    
    return True


if __name__ == "__main__":
    test_fly_detection()
