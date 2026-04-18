# hardware/base_controller.py
from abc import ABC, abstractmethod

class BaseController(ABC):
    """Interface chung — mọi controller đều implement."""

    @abstractmethod
    def send_timing(self, green_ns: int, yellow_ns: int,
                    green_ew: int, yellow_ew: int) -> bool:
        """Gửi bộ thời gian đèn. Trả về True nếu thành công."""
        pass

    @abstractmethod
    def get_status(self) -> dict:
        """Lấy trạng thái hiện tại của hệ thống đèn."""
        pass

    def apply_chromosome(self, chromosome: list) -> bool:
        """Tiện ích: áp dụng trực tiếp từ chromosome [g_ns,y_ns,g_ew,y_ew]."""
        if len(chromosome) < 4:
            return False
        g_ns, y_ns, g_ew, y_ew = chromosome[:4]
        return self.send_timing(g_ns, y_ns, g_ew, y_ew)
