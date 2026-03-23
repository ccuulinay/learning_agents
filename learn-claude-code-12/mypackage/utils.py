"""
Utility functions for common operations.
"""

import re
from typing import Any, List, Optional


def is_valid_email(email: str) -> bool:
    """
    Check if a string is a valid email address.
    
    Args:
        email: The email address to validate.
        
    Returns:
        True if valid, False otherwise.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def reverse_string(text: str) -> str:
    """
    Reverse a given string.
    
    Args:
        text: The string to reverse.
        
    Returns:
        The reversed string.
    """
    return text[::-1]


def find_duplicates(items: List[Any]) -> List[Any]:
    """
    Find duplicate items in a list.
    
    Args:
        items: A list of items to check.
        
    Returns:
        A list of duplicate items (each appears once).
    """
    seen = set()
    duplicates = set()
    for item in items:
        if item in seen:
            duplicates.add(item)
        seen.add(item)
    return list(duplicates)


def safe_get(data: dict, key: str, default: Any = None) -> Any:
    """
    Safely get a value from a dictionary.
    
    Args:
        data: The dictionary to search.
        key: The key to look for.
        default: Default value if key not found.
        
    Returns:
        The value if found, otherwise the default.
    """
    return data.get(key, default)


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks of specified size.
    
    Args:
        items: The list to chunk.
        chunk_size: Size of each chunk.
        
    Returns:
        A list of chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
