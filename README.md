# 🚦 Hệ thống Quản lý Giao thông Thông minh (NCKH 2.0)
Hệ thống tích hợp AI (PPO 2.0) và Thuật toán di truyền (GA) để tối ưu hóa đèn giao thông dựa trên dữ liệu Vision từ Digital Twin.

## 🚀 Cách vận hành nhanh (Khuyên dùng)
Để thuận tiện nhất, bạn hãy sử dụng bảng điều khiển trung tâm:
```powershell
python gui_launcher.py
```
Từ đây, bạn có thể thực hiện mọi tác vụ: Vẽ ROI, Huấn luyện AI, Chạy Digital Twin Dashboard và Xuất biểu đồ báo cáo.

---

## 🛠️ Quy trình Huấn luyện & Mô phỏng (Chi tiết)

### 1. Huấn luyện PPO 2.0 (Dữ liệu ngẫu nhiên - Robust)
Để huấn luyện bộ não AI có khả năng xử lý mọi luồng xe phức tạp:
```powershell
python rl/train_ppo2.0.py
```
- **Model:** Lưu tại `rl/models/ppo_random/ppo_traffic_random_v2.zip`
- **Logs:** Lưu tại `rl/logs2.0/` (Dùng TensorBoard để xem).

### 2. Tối ưu hóa chu kỳ đèn bằng GA
Thuật toán di truyền tìm kiếm bộ thông số "Vàng" cho ngã tư dựa trên luồng xe hiện tại:
```powershell
python simulation/main_ga.py
```
- **Tính năng mới:** Sau khi chạy xong, chương trình tự động cập nhật bộ số tốt nhất vào Dashboard (`demo_live_twin_v2.py`) và lưu lịch sử hội tụ vào `data/ga_history.csv`.

### 3. Digital Twin Dashboard (Bản v2)
Giao diện giám sát thời gian thực, cho phép so sánh trực tiếp PPO vs GA vs Fixed-time:
```powershell
python demo_live_twin_v2.py
```

### 4. Chạy Benchmark & Xuất số liệu (Cho bài báo)
Module thi đấu công bằng để lấy số liệu khoa học (Chạy 300 giây/lượt, 10 lượt):
```powershell
python benchmark/run_comparison.py --trials 10
```
- **Báo cáo:** Kết quả tự động xuất ra file Markdown tại `benchmark/results/report.md`.
- **Biểu đồ:** Sử dụng Launcher để trích xuất `Fitness Curve (GA)` và `Learning Curve (PPO)` với tông màu Premium.

---

## 📟 Triển khai Phần cứng (Raspberry Pi + Arduino)
1. **Arduino:** Nạp file `hardware/arduino/traffic_light.ino` để điều khiển đèn LED vật lý.
2. **Dashboard:** Khi chạy Dashboard, hệ thống sẽ tự động tìm cổng COM của Arduino để đồng bộ tín hiệu đèn thật.

## 📊 Cấu trúc Reward PPO 2.0
Hàm thưởng (Reward) được thiết kế để AI cân bằng giữa việc giải tỏa xe và tránh chuyển đèn quá liên tục:
`Reward = (Speed_Avg * W1) - (Waiting_Time * W2) - (Time_Loss * W3) - (Switch_Penalty * W4)`

## 📊 Các thuật toán tối ưu (Core Logic)

### 🧬 Hàm Fitness (Genetic Algorithm)
GA đánh giá mỗi bộ chu kỳ đèn dựa trên chỉ số Fitness tổng hợp (Càng cao càng tốt). Công thức được thiết lập để cân bằng giữa tốc độ và sự ùn tắc:

$$Fitness = (0.15 \times \text{Speed}) - (0.35 \times \text{TimeLoss}) - (0.35 \times \text{WaitingTime}) - (0.15 \times \text{Density})$$

*Trong đó:*
- **Speed:** Tốc độ trung bình (m/s).
- **TimeLoss/WaitingTime:** Thời gian trễ và thời gian dừng chờ (s).
- **Density:** Mật độ xe trên làn (xe/km).

---

### 🧠 Hàm Reward (PPO Deep RL)
PPO học cách ra quyết định thông qua phản hồi từ môi trường. Hàm thưởng được thiết kế để triệt tiêu các hành vi gây ùn tắc:

$$Reward = - \left( \sum \text{HaltingVehicles} \times 1.0 + \sum \frac{\text{WaitingTime}}{100} \times 0.5 \right) - \text{ActionPenalty} - \text{PhaseChangePenalty}$$

*Cơ chế Reward:*
- **HaltingVehicles:** Tổng số xe đang dừng hẳn tại ngã tư.
- **WaitingTime:** Tổng thời gian chờ tích lũy của các xe.
- **ActionPenalty (=-2):** Hình phạt khi AI "yêu cầu" đổi đèn.
- **PhaseChangePenalty (=-10):** Hình phạt nặng khi hệ thống **thực sự chuyển pha**, giúp AI học cách giữ đèn xanh lâu hơn để giải tỏa xe hiệu quả thay vì đổi đèn liên tục.

---
*Developed for Scientific Research on AI-based Traffic Management Systems.*