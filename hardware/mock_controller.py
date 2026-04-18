# hardware/mock_controller.py
"""Giả lập Pi5 Controller — chạy hoàn toàn trên PC, in ra console."""
from hardware.base_controller import BaseController

class MockController(BaseController):
    def __init__(self):
        self._current = {'green_ns': 30, 'yellow_ns': 4,
                         'green_ew': 30, 'yellow_ew': 4}
        print("[MockController] Khoi tao OK — chay tren PC (khong can Pi5)")

    def send_timing(self, green_ns, yellow_ns, green_ew, yellow_ew) -> bool:
        self._current = dict(green_ns=green_ns, yellow_ns=yellow_ns,
                             green_ew=green_ew, yellow_ew=yellow_ew)
        print(f"[MockController] SET: NS={green_ns}s/{yellow_ns}s | "
              f"EW={green_ew}s/{yellow_ew}s  -->  ACK:OK (simulated)")
        return True

    def get_status(self) -> dict:
        return {'status': 'mock_running', **self._current}
