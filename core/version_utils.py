"""
Utility functions for reading version information from Windows executables.
"""

import os
import sys
from typing import Optional

try:
    import win32api
    HAS_WIN32API = True
except ImportError:
    HAS_WIN32API = False


def get_exe_version(exe_path: str) -> Optional[str]:
    """
    Read version information from a Windows executable file.

    Args:
        exe_path: Path to the .exe file

    Returns:
        Version string (e.g., "1.0.0") or None if version cannot be read
    """
    if not os.path.exists(exe_path):
        return None

    if not HAS_WIN32API:
        # Fallback: If win32api is not available, return None
        # In production, pywin32 should be installed
        return None

    try:
        info = win32api.GetFileVersionInfo(exe_path, "\\")
        ms = info['FileVersionMS']
        ls = info['FileVersionLS']
        version = f"{win32api.HIWORD(ms)}.{win32api.LOWORD(ms)}.{win32api.HIWORD(ls)}"
        return version
    except Exception:
        return None


def get_current_exe_version() -> Optional[str]:
    """
    Get the version of the currently running executable.

    Returns:
        Version string or None if not running as compiled executable
    """
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        exe_path = sys.executable
        return get_exe_version(exe_path)
    return None


def get_main_app_version(installation_dir: Optional[str] = None) -> Optional[str]:
    """
    Get the version of the main application executable.

    Args:
        installation_dir: Directory where the main app is installed.
                         If None, assumes current directory or tries to find it.

    Returns:
        Version string or None if cannot be determined
    """
    # Try to find executable in the expected location
    if installation_dir is None:
        # Try current directory first
        installation_dir = os.path.dirname(os.path.abspath(__file__))

    possible_names = ["ocr_comparison.exe", "orahvision.exe", "reader.exe"]

    for exe_name in possible_names:
        exe_path = os.path.join(installation_dir, exe_name)
        if os.path.exists(exe_path):
            version = get_exe_version(exe_path)
            if version:
                return version

    return None


def get_helper_app_version(installation_dir: Optional[str] = None) -> Optional[str]:
    """
    Get the version of the helper application executable.

    Args:
        installation_dir: Directory where helper app is installed.
                         If None, assumes current directory or tries to find it.

    Returns:
        Version string or None if cannot be determined
    """
    if installation_dir is None:
        installation_dir = os.path.dirname(os.path.abspath(__file__))

    possible_names = ["helper.exe", "updater.exe", "orahvision_helper.exe"]

    for exe_name in possible_names:
        exe_path = os.path.join(installation_dir, exe_name)
        if os.path.exists(exe_path):
            version = get_exe_version(exe_path)
            if version:
                return version

    return None
