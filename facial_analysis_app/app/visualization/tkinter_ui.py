# --------------------------
# overlay.py (now tkinter_ui.py)
# --------------------------
import tkinter as tk
from tkinter import ttk

AU_LEGEND = {
    "au_1": "Inner Brow Raiser",
    "au_2": "Outer Brow Raiser",
    "au_4": "Brow Lowerer",
    "au_6": "Cheek Raiser",
    "au_7": "Lid Tightener",
    "au_10": "Upper Lip Raiser",
    "au_12": "Lip Corner Puller",
    "au_14": "Dimpler",
    "au_15": "Lip Corner Depressor",
    "au_17": "Chin Raiser",
    "au_23": "Lip Tightener",
    "au_24": "Lip Pressor"
}

class AUWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Facial Action Units - Real Time")
        self.root.geometry("800x600")

        self.bars = {}
        self.value_labels = {}

        # AU section
        self.bar_frame = tk.Frame(self.root)
        self.bar_frame.pack(side="top", fill="both", expand=True)

        for au in AU_LEGEND:
            row = tk.Frame(self.bar_frame)
            row.pack(fill="x", pady=4)

            label = tk.Label(row, width=20, text=f"{au.upper()} ({AU_LEGEND[au]})", anchor='w')
            label.pack(side="left")

            bar = ttk.Progressbar(row, length=300, maximum=5.0, mode='determinate')
            bar.pack(side="left", padx=10)
            self.bars[au] = bar

            value_label = tk.Label(row, width=6, text="0.00")
            value_label.pack(side="left")
            self.value_labels[au] = value_label

        # Legend section
        self.legend = tk.Text(self.root, height=6, wrap="word")
        self.legend.pack(side="bottom", fill="x")
        self.legend.insert("1.0", "AU Legend:\n")
        for k, v in AU_LEGEND.items():
            self.legend.insert("end", f"{k.upper()}: {v}\n")
        self.legend.config(state="disabled")

    def update_aus(self, au_data):
        for au in AU_LEGEND:
            intensity_key = f"{au}_intensity"
            binary_key = au

            intensity = au_data.get(intensity_key, 0)
            is_present = au_data.get(binary_key, 0)

            self.bars[au]["value"] = intensity
            self.value_labels[au]["text"] = f"{intensity:.2f}"

            style_name = f"{au}.Horizontal.TProgressbar"
            style = ttk.Style()
            style.theme_use("default")
            style.configure(style_name, troughcolor='gray',
                            background='green' if is_present else 'red')
            self.bars[au].config(style=style_name)

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    win = AUWindow()
    win.run()
