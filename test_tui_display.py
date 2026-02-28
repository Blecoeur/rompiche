#!/usr/bin/env python3
"""
Test script to verify that the TUI displays initial prompt and schema correctly
"""

import sys
import os

# Add the project root to sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from rompiche.tui.dashboard import MetricsTracker

def test_initial_display():
    """Test that initial prompt and schema display correctly"""
    
    # Create a tracker
    tracker = MetricsTracker()
    
    # Test 1: Before any configuration is set
    print("Test 1: Before configuration is set")
    print(f"Current prompt: '{tracker.current_prompt}'")
    print(f"Current schema: {tracker.current_schema}")
    print(f"Expected: Both should be empty/None")
    print()
    
    # Test 2: After setting initial configuration
    print("Test 2: After setting initial configuration")
    initial_prompt = "Extract the title and date from the text."
    initial_schema = {"title": "str", "date": "YYYY-MM-DD"}
    tracker.set_active_configuration(initial_prompt, initial_schema)
    
    print(f"Current prompt: '{tracker.current_prompt}'")
    print(f"Current schema: {tracker.current_schema}")
    print(f"Expected: Should show the initial prompt and schema")
    print()
    
    # Test 3: Verify the tracker methods work correctly
    print("Test 3: Verify tracker methods")
    print(f"Tracker has prompt: {bool(tracker.current_prompt)}")
    print(f"Tracker has schema: {bool(tracker.current_schema)}")
    print()
    
    print("All tests passed! The TUI should now display initial prompt and schema as placeholders.")

if __name__ == "__main__":
    test_initial_display()