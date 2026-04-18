# 🚦 Tối ưu đèn giao thông bằng Thuật toán Di truyền (GA) & Deep Reinforcement Learning (PPO)

Dự án này là giải pháp toàn diện dùng trí tuệ nhân tạo để tối ưu hóa thời gian phát sáng của các pha đèn giao thông tại một ngã tư, dựa trên dữ liệu mô phỏng từ SUMO. Mọi thành phần đều được thiết kế theo dạng Module độc lập, tạo nền tảng vững chắc cho việc nhúng thẳng lên thiết bị IoT ngoài đời thực (Raspberry Pi 5 + Arduino).

## 📁 Cấu trúc dự án

Dự án được chia thành các phân khu chuyên biệt:
- `simulation/`: Module chứa thư viện kết nối SUMO, bộ tối ưu Thuật toán Di truyền (GA) và Trình tạo phòng tập (Gym Env) cho Reinforcement Learning.
- `rl/`: Module chứa bộ não AI sử dụng thuật toán Deep RL PPO tiên tiến.
- `benchmark/`: Kịch bản so sánh đối chiếu thời gian và hiệu năng giữa đèn Cố định, GA và PPO.
- `hardware/`: Mã nguồn đã được chuẩn bị sẵn cho phần cứng nhúng. (Raspberry Pi và Arduino Firmware).
- `data/`: Dữ liệu mạng lưới giao thông tĩnh cho phần mềm SUMO.

## 🧮 Hàm Đánh giá Tối ưu & Thưởng (Fitness & Reward)

Hệ thống sử dụng hai cơ chế đánh giá khác nhau cho hai thuật toán:

### 1. Thuật toán Di truyền (GA - Fitness)
Điểm Fitness được tính dựa trên dữ liệu thống kê sau mỗi lượt mô phỏng (càng lớn càng tốt, tiệm cận tới 0):

$$Fitness = -(0.35 \cdot TL + 0.35 \cdot WT + 0.15 \cdot D - 0.15 \cdot S)$$

* **TL (timeLoss):** Tổng thời gian trễ của phương tiện so với vận tốc lý tưởng (Trọng số: 35%)
* **WT (waitingTime):** Tổng thời gian phương tiện phải dừng chờ (Trọng số: 35%)
* **D (density):** Mật độ phương tiện trên các lane (Trọng số: 15%)
* **S (speed):** Tốc độ trung bình giải tỏa (Trọng số: 15% - Điểm cộng)

### 2. Deep Reinforcement Learning (PPO - Reward)
Phần thưởng được tính toán **thời gian thực** sau mỗi bước nhảy 5 giây:

$$Reward = -(Queue + 0.5 \cdot |\frac{Wait}{100}|)$$

* **Queue:** Tổng số lượng xe đang bị ùn tắc (Halt number) tại 4 nhánh vào.
* **Wait:** Tổng thời gian chờ tích lũy của các xe tại ngã tư (đã chuẩn hóa).
* *Mục tiêu: AI sẽ tối ưu để giảm thiểu đồng thời cả số xe dừng và thời gian chờ.*

---

## 🚀 Hướng dẫn Cài đặt & Chạy dự án

**1. Khởi tạo môi trường ảo an toàn (trên nền Windows)**
Mở Terminal / Powershell trong thư mục này và gõ:
```powershell
py -m venv venv
.\venv\Scripts\activate
```

**2. Cài đặt toàn bộ thư viện & Gói đồ hoạ**
```powershell
pip install -r requirements.txt
pip install tensorboard tqdm rich
```

### 🖥️ Cách 0: Dùng Giao diện Trực quan (Khuyên dùng)
Dự án được trang bị sẵn một bảng điều khiển trung tâm bằng GUI để bạn không cần gõ lệnh. Nó có chứa toàn bộ chức năng (Training, Benchmark, Xuất ảnh biểu đồ). 
Chỉ cần chạy lệnh sau (hoặc click đúp vào file):
```powershell
python gui_launcher.py
```

### 🏁 Cách 1: Huấn luyện bằng Thuật toán tiến hoá (GA)
Thuật toán cũ, tìm kiếm trên diện rộng và đảm bảo tìm được thời lượng cố định tối ưu.
```powershell
python simulation/main_ga.py
```

### 🧠 Cách 2: Huấn luyện bằng AI Học Tăng Cường (PPO Deep RL)
Thuật toán AI sử dụng mạng Neural Network tiên tiến để điều khiển đèn thời gian thực qua TraCI.
* **Hiển thị đèn (GUI):** Để xem đèn nhảy trực tiếp trong lúc học, mở `rl/train_ppo.py` và sửa biến `USE_GUI = True`.
* Mở **Terminal 1** để kích hoạt đồ thị theo dõi:
  ```powershell
  .\venv\Scripts\activate
  tensorboard --logdir rl/logs
  ```
  Truy cập vào trình duyệt web tại địa chỉ: `http://localhost:6006`

* Mở **Terminal 2** để khởi động AI tập luyện:
  ```powershell
  .\venv\Scripts\activate
  python rl/train_ppo.py
  ```

### 📊 Cách 3: Chạy Benchmark So Sánh Đối Chiếu
Module này cho chạy thi đấu công bằng giữa các phương pháp.
```powershell
python benchmark/run_comparison.py --trials 10
python benchmark/visualize_results.py
```

### 📟 Cách 4: Triển khai Phần cứng Giao thông (Raspberry Pi 5 + Arduino)
Hệ thống hỗ trợ chạy mô hình thật thông qua Raspberry Pi làm trung tâm điều khiển (Server) và mạch Arduino điều khiển hệ thống đèn LED vật lý.

1. **Nạp Firmware cho Cổng đèn:** 
   Mở Arduino IDE, nạp file `hardware/arduino/traffic_light.ino` vào mạch vi điều khiển của bạn.
   
2. **Kiểm thử logic trên PC:** 
   Bạn có thể kiểm tra tín hiệu phần mềm xử lý ra sao trước khi lắp điện bằng lệnh:
   ```powershell
   python tests/test_hardware_mock.py
   ```
   
3. **Chạy Server Điều Khiển Chính (Raspberry Pi):** 
   Cắm cáp USB giữa Pi và Arduino, kích hoạt môi trường ảo `venv` và chọn 1 trong 3 chế độ hoạt động cực kỳ linh hoạt sau:
   
   * **Chế độ Điển hình AI Động (PPO - Dynamic):** AI nạp file `ppo_traffic_dynamic.zip`, đóng vai cảnh sát phân luồng liên tục đọc camera thật mỗi 5 giây để ép hướng đi.
     ```bash
     python hardware/pi_controller.py --mode ppo
     ```
   * **Chế độ Tính trước Dài hạn (GA - Static):** Máy tính sẽ đọc kịch bản phân luồng tốt nhất trong file tĩnh `best_chromosome_ga.json` và gán xuống cho đèn chạy suốt cả ngày.
     ```bash
     python hardware/pi_controller.py --mode ga
     ```

### 💎 Cách 5: Chế độ Digital Twin (Khuyên dùng cho Demo)
Đây là tính năng cao cấp nhất, kết hợp sức mạnh của mô phỏng ảo và thiết bị thật. AI PPO sẽ nhìn vào màn hình SUMO GUI và **điều khiển song song** cả xe ảo lẫn bóng đèn thật trên Arduino.

* **Chạy Demo Bản sao số:**
  ```powershell
  python demo_live_twin.py
  ```
  *Lưu ý: Chế độ này yêu cầu bạn đã nạp file `traffic_light.ino` vào Arduino và cắm cáp USB.*

---

*Lưu ý quan trọng: Code controller phần cứng đã hoàn hiện. Tuy nhiên để chế độ PPO nhận được số lượt xe trên thực tế thay vì bộ đếm ảo, bạn cần bổ sung Camera phần cứng vào Hệ thống (sử dụng YOLO tích hợp) thuộc lộ trình ngoài của văn bản hướng dẫn này.*