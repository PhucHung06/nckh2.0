import tkinter as tk

# Input: [NS xanh, NS vàng, EW xanh, EW vàng]
timings = [10, 3, 10, 3]

phases = [
    ("NS", "green", timings[0]),
    ("NS", "yellow", timings[1]),
    ("EW", "green", timings[2]),
    ("EW", "yellow", timings[3]),
]

class TrafficSim:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Light Simulation")

        self.canvas = tk.Canvas(root, width=400, height=400, bg="black")
        self.canvas.pack()

        # Vẽ đèn
        self.lights = {
            "north": self.create_light(180, 50),
            "south": self.create_light(180, 300),
            "east": self.create_light(300, 180),
            "west": self.create_light(50, 180),
        }

        # Bộ đếm (giống LED)
        self.timer_text = self.canvas.create_text(
            200, 200, text="00", fill="cyan", font=("Courier", 40, "bold")
        )

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
            self.canvas.itemconfig(light[c], fill="grey")

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

    def update(self):
        self.update_lights()

        self.canvas.itemconfig(
            self.timer_text,
            text=str(self.remaining).zfill(2)
        )

        self.remaining -= 1

        if self.remaining < 0:
            self.phase_index = (self.phase_index + 1) % len(phases)
            self.remaining = phases[self.phase_index][2]

        self.root.after(1000, self.update)


if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficSim(root)
    root.mainloop()