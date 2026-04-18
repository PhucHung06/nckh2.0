# tests/test_hardware_mock.py
"""Test luồng GA/RL -> Hardware với MockController trên PC."""
import sys
import os

# Them thu muc goc vao path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from hardware.mock_controller import MockController

def test_mock_controller():
    print("--- Testing MockController ---")
    ctrl = MockController()
    
    # Test send_timing
    assert ctrl.send_timing(30, 4, 45, 4) == True
    
    # Test apply_chromosome
    assert ctrl.apply_chromosome([45, 4, 35, 4]) == True
    
    # Test get_status
    status = ctrl.get_status()
    assert 'green_ns' in status
    assert status['green_ns'] == 45
    
    print("PASSED — MockController hoat dong dung")

if __name__ == '__main__':
    test_mock_controller()
