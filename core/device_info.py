"""
Device information collection module for OCR Comparison tool.

This module collects device-specific information used for client-server
communication headers, including:
- Device license key
- Software version
- API version
- Windows machine ID (fingerprint)
"""

import hashlib
import json
import os
import subprocess
import uuid
from pathlib import Path
from typing import Dict, Optional

try:
    import winreg
    HAS_WINREG = True
except ImportError:
    HAS_WINREG = False

from version import SOFTWARE_VERSION, API_VERSION, HELPER_VERSION
from version_utils import get_main_app_version, get_helper_app_version


# Configuration file location
CONFIG_DIR = Path(os.environ.get('PROGRAMDATA', 'C:\\ProgramData')) / 'Orahvision'
CONFIG_FILE = CONFIG_DIR / 'device.conf'


def get_machine_guid() -> Optional[str]:
    """
    Get Windows MachineGuid from registry.

    Returns:
        MachineGuid string or None if cannot be retrieved
    """
    if not HAS_WINREG:
        return None

    try:
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r'SOFTWARE\Microsoft\Cryptography',
            0,
            winreg.KEY_READ | winreg.KEY_WOW64_64KEY
        )
        machine_guid, _ = winreg.QueryValueEx(key, 'MachineGuid')
        winreg.CloseKey(key)
        return machine_guid
    except Exception:
        return None


def get_hardware_uuid() -> Optional[str]:
    """
    Get hardware UUID using WMIC.

    Returns:
        Hardware UUID string or None if cannot be retrieved
    """
    try:
        result = subprocess.run(
            ['wmic', 'csproduct', 'get', 'UUID'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 2:
                uuid_value = lines[1].strip()
                return uuid_value if uuid_value else None
    except Exception:
        pass
    return None


def get_cpu_id() -> Optional[str]:
    """
    Get CPU processor ID.

    Returns:
        CPU ID string or None if cannot be retrieved
    """
    try:
        result = subprocess.run(
            ['wmic', 'cpu', 'get', 'ProcessorId'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 2:
                cpu_id = lines[1].strip()
                return cpu_id if cpu_id else None
    except Exception:
        pass
    return None


def get_hdd_serial() -> Optional[str]:
    """
    Get primary hard drive serial number.

    Returns:
        HDD serial number or None if cannot be retrieved
    """
    try:
        result = subprocess.run(
            ['wmic', 'diskdrive', 'get', 'SerialNumber'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 2:
                serial = lines[1].strip()
                return serial if serial else None
    except Exception:
        pass
    return None


def get_device_fingerprint() -> str:
    """
    Generate a unique device fingerprint by combining multiple hardware identifiers.

    The fingerprint is a SHA-256 hash of:
    - MachineGuid
    - Hardware UUID
    - CPU ID
    - HDD Serial

    Returns:
        64-character hex string (SHA-256 hash)
    """
    components = []

    machine_guid = get_machine_guid()
    if machine_guid:
        components.append(f"MG:{machine_guid}")

    hw_uuid = get_hardware_uuid()
    if hw_uuid:
        components.append(f"HW:{hw_uuid}")

    cpu_id = get_cpu_id()
    if cpu_id:
        components.append(f"CPU:{cpu_id}")

    hdd_serial = get_hdd_serial()
    if hdd_serial:
        components.append(f"HDD:{hdd_serial}")

    # Combine all components
    fingerprint_input = "|".join(components)

    # Generate SHA-256 hash
    hash_object = hashlib.sha256(fingerprint_input.encode('utf-8'))
    return hash_object.hexdigest()


def generate_license_key() -> str:
    """
    Generate a new device license key.

    Format: OV628-XXXX-XXXX-XXXX-XXXX
    Uses UUID4 for randomness.

    Returns:
        Formatted license key string
    """
    # Generate UUID4
    key_uuid = uuid.uuid4()
    hex_string = key_uuid.hex.upper()

    # Split into 4-character segments
    segments = [hex_string[i:i+4] for i in range(0, 16, 4)]

    # Format as OV628-XXXX-XXXX-XXXX-XXXX
    return f"OV628-{'-'.join(segments)}"


def load_config() -> Dict:
    """
    Load device configuration from file.

    Returns:
        Configuration dictionary (empty dict if file doesn't exist)
    """
    if not CONFIG_FILE.exists():
        return {}

    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def save_config(config: Dict) -> None:
    """
    Save device configuration to file.

    Args:
        config: Configuration dictionary to save
    """
    # Create directory if it doesn't exist
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        # Log error but don't crash
        print(f"Warning: Could not save config: {e}")


def get_or_create_license_key() -> str:
    """
    Get existing license key from config, or generate and save a new one.

    Returns:
        License key string
    """
    config = load_config()

    if 'license_key' in config and config['license_key']:
        return config['license_key']

    # Generate new license key
    license_key = generate_license_key()
    config['license_key'] = license_key
    save_config(config)

    return license_key


def get_device_info(installation_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Get complete device information for client-server communication.

    This is the main API function that collects all device information.

    Args:
        installation_dir: Optional installation directory for finding executables

    Returns:
        Dictionary containing:
        - device_license_key: Unique device license key
        - software_version: Main application version
        - api_version: API version client expects
        - windows_id: Device fingerprint
    """
    # Try to get version from compiled executable, fallback to hardcoded
    software_version = get_main_app_version(installation_dir) or SOFTWARE_VERSION

    device_info = {
        'device_license_key': get_or_create_license_key(),
        'software_version': software_version,
        'api_version': API_VERSION,
        'windows_id': get_device_fingerprint(),
    }

    return device_info


def get_helper_device_info(installation_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Get device information from helper app perspective.

    Includes helper app version in addition to standard device info.

    Args:
        installation_dir: Optional installation directory for finding executables

    Returns:
        Dictionary containing all device info plus helper_version
    """
    device_info = get_device_info(installation_dir)

    # Try to get helper version from compiled executable, fallback to hardcoded
    helper_version = get_helper_app_version(installation_dir) or HELPER_VERSION
    device_info['helper_version'] = helper_version

    return device_info


if __name__ == '__main__':
    # Test the module
    print("Device Information:")
    print("-" * 50)

    info = get_device_info()
    for key, value in info.items():
        print(f"{key}: {value}")
