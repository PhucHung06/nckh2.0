# Kế hoạch Tái cấu trúc Train PPO - Điều khiển Động bằng Traci (Lựa chọn 2)

Dự án sẽ chuyển đổi mô hình PPO từ việc tối ưu hóa tĩnh (1 lượt đánh giá = 1 cấu hình) sang **Điều khiển động thời gian thực**, trong đó AI giống như một cảnh sát giao thông, liên tục theo dõi tình hình ngã tư và đưa ra quyết định dựa trên số lượng xe đang chờ.

## User Review Required

> [!WARNING]
> Việc đập bỏ hoàn toàn cơ chế cũ đối với PPO đồng nghĩa với việc AI sẽ không sinh ra một "cấu hình đèn tĩnh cố định" nữa (như `[30, 4, 30, 4]` của GA). 
> Thay vào đó, AI học sinh ra một **Policy (Chính sách)**, tức là nhận đầu vào (số xe) và cho ra đầu ra (giữ đèn phân luồng này hay chuyển sang đèn phân luồng khác). 
> Việc export model sang phần cứng (Raspberry Pi/Arduino) sau này sẽ yêu cầu một script chạy inference bằng Python (load file `.zip`) thay vì chỉ đọc file `json` tĩnh. Bạn xác nhận ổn với điều này chứ?

## Proposed Changes

### Thay đổi Kiến Trúc Môi trường (Gymnasium)

Thiết kế lại vòng lặp của môi trường: Tần suất AI ra quyết định sẽ là mỗi **5 giây mô phỏng**. (Nghĩa là cứ 5s, AI lại được hỏi có muốn chuyển đèn không).

#### [MODIFY] `simulation/sumo_gym_env.py`
Xóa toàn bộ liên kết với `sumo_env.py` (file cũ này chỉ giữ lại cho `main_ga.py`). Dùng trực tiếp `traci` và `libsumo` (tối ưu tốc độ) để code môi trường.
- **Action Space (`Discrete(2)`)**: 
  - `0`: Kéo dài đèn xanh hiện tại thêm 5 giây.
  - `1`: Chuyển sang chu kỳ đèn tiếp theo (Ví dụ: Đang Xanh NS -> Vàng NS (4s) -> Xanh EW).
  - *Luật bắt buộc: Đèn xanh cần sáng tối thiểu 10 giây mới cho phép AI chuyển đèn (để tránh xe bị nhấp nháy).*
- **Observation Space (`Box(8,)`)**:
  - Chiều dài hàng chờ (Queue length) ở 4 ngõ vào.
  - Tổng thời gian chờ (Accumulated waiting time) ở 4 ngõ vào.
  - Thông tin này cung cấp "tầm nhìn" cho AI.
- **Reward Function**: 
  - Thưởng/Phạt = Độ thay đổi của tổng delay giữa bước trước và bước này (Giảm delay = Thưởng dương, Tăng delay = Phạt âm).

#### [MODIFY] `rl/train_ppo.py`
Điều chỉnh lại training loop để phù hợp với môi trường Traci.
- **Tập dữ liệu**: Số bước mô phỏng sẽ tăng mạnh (một lần chạy 1 giờ mất khoảng 720 bước AI ra quyết định). Nâng `total_timesteps` lên `100,000` hoặc cao hơn. Traci chạy rất nhanh.
- **Xóa phần sinh file JSON**: Thay vì ghi đè file `best_chromosome_rl.json`, ta sẽ sinh file model `.zip` để tải vào các thiết bị sau.

#### [NEW] Cấu hình Net XML hỗ trợ Traci (Tùy chọn)
- Nếu file `ngatu.net.xml` hiện tại đang sử dụng phase fix cứng 24 ký tự State. Traci hỗ trợ gọi qua API `traci.trafficlight.setRedYellowGreenState("Center", "GGGGggrrrrrr...").` Ta không cần can thiệp file XML nữa mà ghi đè bằng code trong Python.

## Open Questions

- Bạn cài đặt thư viện `traci` cho python chưa? Nếu chưa, lệnh là `pip install sumolib traci`.
- Phần cứng bạn định kết nối (Ras Pi 5) có khả năng chạy code Python nạp model `stable-baselines3` không? Nếu không, mình có thể thêm bước xuất mô hình RL ra dạng ONNX (rất nhẹ để chạy trên Pi).

## Verification Plan
1. Viết xong `sumo_gym_env.py` mới, sẽ dùng random action để test Traci có khởi động SUMO UI/CMD và đèn có đổi màu khớp hay không.
2. Thiết lập training khoảng 10,000 steps và kiểm tra `tensorboard` xem AI có giảm thiểu được hàng chờ không.
