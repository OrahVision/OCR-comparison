"""
API utilities for credential management
"""

import logging
import sys
from pathlib import Path

# Add core directory to path to import device_info
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

try:
    from device_info import get_device_info
    HAS_DEVICE_INFO = True
except ImportError:
    HAS_DEVICE_INFO = False
    logging.error("device_info module not found")


class UserCredentials:
    """Manages user credentials from device_info module"""

    def __init__(self):
        """
        Initialize credentials from device_info module

        Gets registration_key (device_license_key) and device_id (windows_id)
        from the device_info module.
        """
        self.registration_key = None
        self.device_id = None

        self._load_credentials()

    def _load_credentials(self):
        """Load credentials from device_info module"""
        if not HAS_DEVICE_INFO:
            logging.error("Cannot load credentials - device_info module not available")
            return

        try:
            device_info = get_device_info()

            # Extract credentials
            self.registration_key = device_info.get('device_license_key')
            self.device_id = device_info.get('windows_id')

            if self.registration_key and self.device_id:
                logging.info("Loaded credentials from device_info module")
            else:
                logging.warning(f"Incomplete credentials from device_info: registration_key={bool(self.registration_key)}, device_id={bool(self.device_id)}")

        except Exception as e:
            logging.error(f"Error loading credentials from device_info: {e}")

    def is_valid(self) -> bool:
        """Check if credentials are valid"""
        return self.registration_key is not None and self.device_id is not None
