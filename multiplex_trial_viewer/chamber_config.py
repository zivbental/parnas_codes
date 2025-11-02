"""
Chamber configuration manager for multiplex trial viewer.
Handles loading, saving, and editing chamber configurations.
"""

import json
import os
from typing import Dict, Any, Optional


class ChamberConfig:
    """
    Manages chamber configuration loading, saving, and validation.
    """
    
    def __init__(self, config_file: str = "chambers_configuration.json"):
        """
        Initialize the chamber configuration manager.
        
        Args:
            config_file (str): Path to the chamber configuration JSON file
        """
        # Convert to absolute path if relative
        if not os.path.isabs(config_file):
            config_file = os.path.abspath(config_file)
        
        self.config_file = config_file
        self.chambers = {}
        self.load_config()
    
    def load_config(self) -> Dict[str, Dict[str, int]]:
        """
        Load chamber configuration from JSON file.
        If file doesn't exist, create default configuration.
        
        Returns:
            Dict[str, Dict[str, int]]: Chamber configuration dictionary
        """
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as file:
                    self.chambers = json.load(file)
                print(f"Loaded chamber configuration from {self.config_file}")
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading chamber config: {e}")
                self._create_default_config()
        else:
            print(f"Chamber config file not found: {self.config_file}")
            self._create_default_config()
        
        return self.chambers
    
    def _create_default_config(self) -> None:
        """
        Create default chamber configuration with 20 chambers.
        """
        self.chambers = {}
        for i in range(1, 21):
            self.chambers[f'chamber_{i}'] = {
                'x': 0,
                'y': 0,
                'width': 0,
                'height': 0
            }
        print("Created default chamber configuration")
    
    def save_config(self) -> bool:
        """
        Save current chamber configuration to JSON file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            import os
            config_dir = os.path.dirname(self.config_file)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir, exist_ok=True)
                print(f"Created directory: {config_dir}")
            
            # Save the configuration
            with open(self.config_file, 'w') as file:
                json.dump(self.chambers, file, indent=4)
            print(f"Saved chamber configuration to {self.config_file}")
            return True
        except IOError as e:
            print(f"Error saving chamber config to {self.config_file}: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error saving chamber config: {e}")
            return False
    
    def get_chamber(self, chamber_id: str) -> Optional[Dict[str, int]]:
        """
        Get configuration for a specific chamber.
        
        Args:
            chamber_id (str): Chamber identifier (e.g., 'chamber_1')
            
        Returns:
            Optional[Dict[str, int]]: Chamber configuration or None if not found
        """
        return self.chambers.get(chamber_id)
    
    def set_chamber(self, chamber_id: str, x: int, y: int, width: int, height: int) -> bool:
        """
        Set configuration for a specific chamber.
        
        Args:
            chamber_id (str): Chamber identifier
            x (int): X coordinate
            y (int): Y coordinate
            width (int): Width
            height (int): Height
            
        Returns:
            bool: True if successful, False otherwise
        """
        if chamber_id not in self.chambers:
            print(f"Chamber {chamber_id} not found")
            return False
        
        self.chambers[chamber_id] = {
            'x': x,
            'y': y,
            'width': width,
            'height': height
        }
        print(f"Updated {chamber_id}: x={x}, y={y}, width={width}, height={height}")
        return True
    
    def get_all_chambers(self) -> Dict[str, Dict[str, int]]:
        """
        Get all chamber configurations.
        
        Returns:
            Dict[str, Dict[str, int]]: All chamber configurations
        """
        return self.chambers.copy()
    
    def validate_config(self) -> bool:
        """
        Validate chamber configuration for completeness and reasonable values.
        
        Returns:
            bool: True if valid, False otherwise
        """
        if not self.chambers:
            print("No chambers configured")
            return False
        
        for chamber_id, config in self.chambers.items():
            required_keys = ['x', 'y', 'width', 'height']
            for key in required_keys:
                if key not in config:
                    print(f"Missing {key} in {chamber_id}")
                    return False
                if not isinstance(config[key], (int, float)):
                    print(f"Invalid {key} type in {chamber_id}")
                    return False
                if config[key] < 0:
                    print(f"Negative {key} value in {chamber_id}")
                    return False
        
        print("Chamber configuration is valid")
        return True
    
    def get_chamber_count(self) -> int:
        """
        Get the number of configured chambers.
        
        Returns:
            int: Number of chambers
        """
        return len(self.chambers)
    
    def reset_to_default(self) -> None:
        """
        Reset all chambers to default configuration (all zeros).
        """
        for chamber_id in self.chambers:
            self.chambers[chamber_id] = {'x': 0, 'y': 0, 'width': 0, 'height': 0}
        print("Reset all chambers to default configuration")


def test_chamber_config():
    """
    Test function for chamber configuration functionality.
    """
    print("Testing chamber configuration...")
    
    # Create a test configuration manager
    config = ChamberConfig("test_chambers.json")
    
    # Test loading
    chambers = config.get_all_chambers()
    print(f"Loaded {len(chambers)} chambers")
    
    # Test setting a chamber
    success = config.set_chamber("chamber_1", 100, 200, 50, 30)
    print(f"Set chamber_1: {success}")
    
    # Test getting a chamber
    chamber_1 = config.get_chamber("chamber_1")
    print(f"Retrieved chamber_1: {chamber_1}")
    
    # Test validation
    is_valid = config.validate_config()
    print(f"Configuration is valid: {is_valid}")
    
    # Test saving
    save_success = config.save_config()
    print(f"Save successful: {save_success}")
    
    # Clean up test file
    if os.path.exists("test_chambers.json"):
        os.remove("test_chambers.json")
        print("Cleaned up test file")
    
    print("Chamber configuration test completed successfully!")
    return True


if __name__ == "__main__":
    test_chamber_config()
