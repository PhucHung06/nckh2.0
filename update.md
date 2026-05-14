# Kế hoạch triển khai: Huấn luyện PPO trên dữ liệu giao thông ngẫu nhiên (train_ppo2.0)

Mục tiêu: Xây dựng một luồng huấn luyện mới hoàn toàn tách biệt, sử dụng dữ liệu sinh ngẫu nhiên liên tục trong 1 giờ để AI (PPO) có thể học được các quy luật giao thông tổng quát nhất, khắc phục nhược điểm "học vẹt" của dữ liệu 1 phút.

## 1. Dữ liệu mô phỏng (Random Traffic Flow)
Thay vì dùng lại dữ liệu cố định từ YOLO (`yolo_routes.rou.xml`), chúng ta sẽ tạo một file định tuyến mới sử dụng thẻ `<flow>` của SUMO. Thẻ này cho phép sinh xe ngẫu nhiên liên tục theo xác suất từ giây thứ 0 đến 3600 (1 tiếng).
- **Tạo file mới:** `data/xml/random_flow.rou.xml` định nghĩa các luồng xe (N-S, E-W, v.v...) với xác suất xuất hiện linh hoạt.
- **Tạo cấu hình mới:** `data/run_random.sumocfg` trỏ tới file `random_flow.rou.xml` và `ngatu.net.xml`.

## 2. Script Huấn luyện (train_ppo2.0.py)
Tạo script `rl/train_ppo2.0.py` dựa trên phiên bản cũ nhưng có những thay đổi cốt lõi:
- **Môi trường (Env):** Trỏ `sumocfg` tới `data/run_random.sumocfg`.
- **Độ dài Episode:** Thay vì `max_steps = 12` hay `20`, ta sẽ để `max_steps = 720` (Tương đương 1 tiếng mô phỏng cho mỗi video/episode) để AI có không gian học tập dài hạn.
- **Thư mục lưu trữ (Log & Model):**
  - Chuyển thư mục tensorboard sang: `rl/logs2.0/`
  - Chuyển thư mục lưu Model sang: `rl/models/ppo_random/`
- **Thời gian huấn luyện:** `total_timesteps` sẽ được đẩy lên mức `100,000` hoặc `200,000` để đảm bảo độ hội tụ trên dữ liệu ngẫu nhiên.

## 3. Quá trình thực thi
1. Viết code tạo file `random_flow.rou.xml` và `run_random.sumocfg`.
2. Khởi tạo `train_ppo2.0.py` với các thông số cấu hình tối ưu nhất cho Random Flow.
3. Chạy lệnh huấn luyện và theo dõi đồ thị trên Tensorboard qua thư mục `logs2.0`.
4. (Tùy chọn) Mang model học được test ngược lại trên `demo_live_twin.py` (dữ liệu 1 phút) để xem hiệu năng.
