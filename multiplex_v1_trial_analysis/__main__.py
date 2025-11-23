"""
Package Entry Point

This module allows the package to be run as a module using:
    python -m multiplex_v1_trial_analysis

When executed this way, it imports and calls the main() function from the
main module, which handles command-line argument parsing and orchestrates
the complete analysis pipeline.
"""

from .main import main

if __name__ == '__main__':
    main()

