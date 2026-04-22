"""A simple Hello World module.

This module provides a function to print a greeting message.
"""


def greet(name: str = "World") -> None:
    """Print a greeting message.

    Args:
        name: The name to include in the greeting. Defaults to "World".
    """
    print(f"Hello, {name}!")


if __name__ == "__main__":
    greet()
