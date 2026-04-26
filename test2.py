import tkinter as tk

# ================== CONFIG ==================
timings = [10, 4, 10, 3]

phases = [
    ("NS", "green", timings[0]),
    ("NS", "yellow", timings[1]),
    ("EW", "green", timings[2]),
    ("EW", "yellow", timings[3]),
]

# ================== LED 7 SEG ==================
SEGMENTS = {
    '0': (1,1,1,1,1,1,0),
    '1': (0,1,1,0,0,0,0),
    '2': (1,1,0,1,1,0,1),
    '3': (1,1,1,1,0,0,1),
    '4': (0,1,1,0,0,1,1),
    '5': (1,0,1,1,0,1,1),
    '6': (1,0,1,1,1,1,1),
    '7': (1,1,1,0,0,0,0),
    '8': (1,1,1,1,1,1,1),
    '9': (1,1,1,1,0,1,1)
}

class SevenSegment:
    def __init__(self, canvas, x, y, size=10):
        self.canvas = canvas
        s = size

        self.segs = [
            canvas.create_rectangle(x+s, y, x+3*s, y+s),
            canvas.create_rectangle(x+3*s, y+s, x+4*s, y+3*s),
            canvas.create_rectangle(x+3*s, y+4*s, x+4*s, y+6*s),
            canvas.create_rectangle(x+s, y+6*s, x+3*s, y+7*s),
            canvas.create_rectangle(x, y+4*s, x+s, y+6*s),
            canvas.create_rectangle(x, y+s, x+s, y+3*s),
            canvas.create_rectangle(x+s, y+3*s, x+3*s, y+4*s)
        ]

    def display(self, num):
        pattern = SEGMENTS.get(num, (0,0,0,0,0,0,0))
        for seg, on in zip(self.segs, pattern):
            color = "#00FFAA" if on else "#022"
            self.canvas.itemconfig(seg, fill=color)

# ================== TRAFFIC ==================
class TrafficSim:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Light + LED Display")

        self.canvas = tk.Canvas(root, width=500, height=500, bg="#111")
        self.canvas.pack()

        # Đèn giao thông
        self.lights = {
            "north": self.create_light(230, 50),
            "south": self.create_light(230, 350),
            "east": self.create_light(350, 230),
            "west": self.create_light(50, 230),
        }

        # LED ở giữa
        self.digit1 = SevenSegment(self.canvas, 180, 200, 12)
        self.digit2 = SevenSegment(self.canvas, 260, 200, 12)

        self.phase_index = 0
        self.remaining = phases[0][2]

        self.update()

    def create_light(self, x, y):
        r = self.canvas.create_oval(x, y, x+30, y+30, fill="grey")
        yel = self.canvas.create_oval(x, y+35, x+30, y+65, fill="grey")
        g = self.canvas.create_oval(x, y+70, x+30, y+100, fill="grey")
        return {"red": r, "yellow": yel, "green": g}

    def set_light(self, light, color):
        for c in ["red", "yellow", "green"]:
            self.canvas.itemconfig(light[c], fill="#222")

        self.canvas.itemconfig(light[color], fill=color)

    def update_lights(self):
        direction, color, _ = phases[self.phase_index]

        if direction == "NS":
            self.set_light(self.lights["north"], color)
            self.set_light(self.lights["south"], color)
            self.set_light(self.lights["east"], "red")
            self.set_light(self.lights["west"], "red")
        else:
            self.set_light(self.lights["east"], color)
            self.set_light(self.lights["west"], color)
            self.set_light(self.lights["north"], "red")
            self.set_light(self.lights["south"], "red")

    def update_led(self):
        s = str(self.remaining).zfill(2)
        self.digit1.display(s[0])
        self.digit2.display(s[1])

    def update(self):
        self.update_lights()
        self.update_led()

        self.remaining -= 1

        if self.remaining < 0:
            self.phase_index = (self.phase_index + 1) % len(phases)
            self.remaining = phases[self.phase_index][2]

        self.root.after(1000, self.update)

# ================== RUN ==================
if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficSim(root)
    root.mainloop()