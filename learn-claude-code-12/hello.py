"""
Hello module - A simple greeting utility.

This module provides a function to greet users by name.
"""


def hello(name: str) -> str:
    """
    Greet someone by name.

    Args:
        name: The name of the person to greet.

    Returns:
        A greeting message addressed to the provided name.

    Example:
        >>> hello("Alice")
        'Hello, Alice!'
    """
    return f"Hello, {name}!"


def main() -> None:
    """
    Main entry point for the script.

    Demonstrates example usage of the hello function.
    """
    # Example usage
    print(hello("World"))


if __name__ == "__main__":
    main()
