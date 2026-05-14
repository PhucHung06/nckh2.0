# hardware/pi_controller.py
"""Controller thật trên Raspberry Pi 5. Drop-in thay thế MockController."""
import serial
import time
import json
import os
import sys

# Thêm thư mục gốc vào path để có thể import từ module 'hardware'
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from hardware.base_controller import BaseController

SERIAL_PORT = '/dev/ttyUSB0'  # Kiểm tra: ls /dev/tty* | grep -E 'USB|ACM'
BAUD_RATE = 9600

class PiController(BaseController):
    def __init__(self, port=None, baud=BAUD_RATE):
        self.ser = None
        ports_to_try = [port] if port else []
        
        # Thêm các cổng dự phòng tùy theo OS
        if os.name == 'nt': # Windows
            ports_to_try += [f'COM{i}' for i in range(1, 21)]
        else: # Linux/Mac
            ports_to_try += ['/dev/ttyUSB0', '/dev/ttyACM0', '/dev/ttyAMA0']
            
        print(f"[PiController] Dang tim kiem cong ket noi Arduino...")
        for p in ports_to_try:
            try:
                self.ser = serial.Serial(p, baud, timeout=1)
                time.sleep(2)
                print(f"[PiController] Da ket noi thanh cong tai cong: {p}")
                return
            except:
                continue
        
        print("[PiController] CANH BAO: Khong tim thay Arduino. He thong se chay o che do offline (No Hardware).")

    def send_timing(self, green_ns, yellow_ns, green_ew, yellow_ew) -> bool:
        if not self.ser:
            return False
        cmd = f"SET:{green_ns}:{yellow_ns}:{green_ew}:{yellow_ew}\n"
        self.ser.write(cmd.encode())
        resp = self.ser.readline().decode().strip()
        ok = (resp == "ACK:OK")
        print(f"[Pi] {cmd.strip()}  -->  {'OK' if ok else 'LOI: ' + resp}")
        return ok

    def get_status(self) -> dict:
        if not self.ser:
            return {"error": "No serial connection"}
        self.ser.write(b"STATUS\n")
        raw = self.ser.readline().decode().strip()
        parts = raw.split(':')
        return {
            'raw': raw,
            'phase': parts[1] if len(parts) > 1 else '?',
            'elapsed_ms': parts[2] if len(parts) > 2 else '?'
        }

def main():
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(description="Dieu khien den giao thong Raspberry Pi")
    parser.add_argument('--mode', type=str, choices=['ppo', 'ga', 'fixed'], default='ppo',
                        help="Che do chay: ppo (Real-time AI), ga (Doc file json), fixed (Co dinh 30s)")
    args = parser.parse_args()

    ctrl = PiController()

    if args.mode == 'ga':
        # Chạy chế độ GA (Tĩnh)
        ga_path = os.path.join(os.path.dirname(__file__), 'config', 'best_chromosome_ga.json')
        if os.path.exists(ga_path):
            try:
                with open(ga_path) as f:
                    data = json.load(f)
                print(f"[Pi - GA Mode] Nap: {data['chromosome']} (method={data.get('method', 'Unknown')})")
                ctrl.apply_chromosome(data['chromosome'])
            except Exception as e:
                print(f"[Pi] Loi doc file config GA: {e}")
        else:
            print("[Pi] Khong tim thay file JSON cua GA. Fallback sang Fixed.")
            ctrl.send_timing(30, 4, 30, 4)

    elif args.mode == 'ppo':
        # Chạy chế độ PPO (Real-time)
        try:
            from stable_baselines3 import PPO
            HAS_SB3 = True
        except ImportError:
            HAS_SB3 = False
            print("[Pi] Canh bao: Khong tim thay thu vien stable-baselines3.")

        MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'rl', 'models', 'ppo_dynamic', 'ppo_traffic_dynamic.zip')
        
        if HAS_SB3 and os.path.exists(MODEL_PATH):
            print(f"[Pi - PPO Mode] Dang nap mo hinh DRL PPO tu {MODEL_PATH}...")
            model = PPO.load(MODEL_PATH, device='cpu')
            print("[Pi] Hoan tat nap mo hinh. Bat dau vong lap dieu khien thoi gian thuc!")
            
            try:
                while True:
                    # Mock dữ liệu thực tế từ module Vision/Camera (YOLO)
                    simulated_obs = np.array([5, 2, 10, 0, 15.0, 5.0, 30.0, 0.0], dtype=np.float32)
                    action, _states = model.predict(simulated_obs, deterministic=True)
                    
                    if action == 1:
                        print(f"[Pi] Nhin thay {simulated_obs[:4]} xe -> AI Quyen dinh: CHUYEN PHA (Switch)")
                        # Gửi lệnh ep chuyển pha hoặc gửi thời gian pha mới
                    else:
                        print(f"[Pi] Nhin thay {simulated_obs[:4]} xe -> AI Quyen dinh: GIU NGUYEN (Stay)")
                        
                    time.sleep(5)  # Chu ky de AI ra quyet dinh (delta_time = 5s)
            except KeyboardInterrupt:
                print("\n[Pi] Da dung chuong trinh dieu khien.")
        else:
            print("[Pi] Khong tim thay mo hinh PPO (.zip). Chuyen sang Fixed mode.")
            ctrl.send_timing(30, 4, 30, 4)
            
    elif args.mode == 'fixed':
        print("[Pi - Fixed Mode] Chay cau hinh co dinh 30/4/30/4.")
        ctrl.send_timing(30, 4, 30, 4)

if __name__ == '__main__':
    main()
