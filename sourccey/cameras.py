"""Simulated camera feeds and a five-camera desktop preview."""
from pathlib import Path
import time

import mujoco
from PIL import Image, ImageDraw


CAMERAS = ("front_left", "front_right", "bottom", "wrist_left", "wrist_right")
FEED_SIZE = (320, 240)
LABEL_HEIGHT = 22


def render_tiled(model, data, renderer):
    """Render the five model cameras into a labeled RGB image."""
    width, height = FEED_SIZE
    image = Image.new("RGB", (width * 3, (height + LABEL_HEIGHT) * 2), (15, 18, 20))
    draw = ImageDraw.Draw(image)
    for index, name in enumerate(CAMERAS):
        renderer.update_scene(data, camera=name)
        tile = Image.fromarray(renderer.render())
        x = (index % 3) * width
        y = (index // 3) * (height + LABEL_HEIGHT)
        image.paste(tile, (x, y + LABEL_HEIGHT))
        draw.text((x + 6, y + 4), name.replace("_", " "), fill=(100, 240, 150))
    return image


def save_snapshot(model, data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(model, height=FEED_SIZE[1], width=FEED_SIZE[0]) as renderer:
        render_tiled(model, data, renderer).save(path)


class CameraFeeds:
    """Optional Tk preview; render at 5 Hz so physics and the main viewer stay responsive."""

    def __init__(self, parent, simulation):
        import tkinter as tk
        from PIL import ImageTk

        self.ImageTk = ImageTk
        self.simulation = simulation
        self.window = tk.Toplevel(parent)
        self.window.title("Sourccey | Simulated camera feeds (provisional poses)")
        self.window.resizable(False, False)
        self.label = tk.Label(self.window)
        self.label.pack()
        self.image = None
        self.next_frame = 0.0
        self.renderer = mujoco.Renderer(simulation.model, height=FEED_SIZE[1], width=FEED_SIZE[0])
        self.window.protocol("WM_DELETE_WINDOW", self.close)

    def update(self):
        if self.window is None or time.monotonic() < self.next_frame:
            return
        self.next_frame = time.monotonic() + 0.2
        frame = render_tiled(self.simulation.model, self.simulation.data, self.renderer)
        self.image = self.ImageTk.PhotoImage(frame, master=self.window)
        self.label.configure(image=self.image)

    def close(self):
        if self.window is not None:
            self.window.destroy()
            self.window = None
            self.renderer.close()
