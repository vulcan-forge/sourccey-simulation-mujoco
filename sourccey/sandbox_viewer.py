"""Native MuJoCo 4 workshop UI and window-local held-key input."""
from dataclasses import dataclass
import time

import mujoco
import numpy as np
from mujoco.studio import app, messages, plugin
from mujoco.studio.dear_imgui import dear_imgui as imgui

from .build import ROLES
from .sandbox import Controls, overview_camera, visible_options, copy_display_state


@dataclass(frozen=True)
class ManualInput(messages.Snapshot):
    controls: Controls


@dataclass(frozen=True)
class WorkshopAction(messages.Event):
    action: str


@dataclass(frozen=True)
class WorkshopStatus(messages.Snapshot):
    sim_time: float
    realtime: float
    paused: bool
    joints: tuple
    ik_error: float
    lifted: int
    floor_boxes: int


@dataclass(frozen=True)
class FeedImage(messages.Snapshot):
    pixels: np.ndarray


class WorkshopApp(app.App):
    """Retain orbit/zoom, replacing Studio's overlapping keyboard shortcuts/UI."""
    def handle_keyboard_events(self):
        pass

    def build_gui(self):
        pass


class WorkshopPhysics(plugin.SimPlugin):
    def __init__(self, workshop, display_model, display_data):
        self.workshop = workshop
        self.display_model, self.display_data = display_model, display_data
        self.last = time.monotonic()
        self.start = self.last
        self.accumulator = 0.
        self.next_status = 0.
        self.feeds = False
        self.renderer = None
        self.next_image = 0.
        self.ratio = 1.

    @messages.handler
    def input(self, event: ManualInput):
        self.workshop.receive(event.controls)

    @messages.handler
    def action(self, event: WorkshopAction):
        w = self.workshop
        if event.action == "reset":
            w.reset()
            self.accumulator = 0.
            self.last = time.monotonic()
        elif event.action == "hold":
            w.hold()
        elif event.action == "pause":
            w.hold()
            w.paused = not w.paused
            self.accumulator = 0.
        elif event.action == "feeds":
            self.feeds = not self.feeds
            if not self.feeds:
                self.close_renderer()
        elif event.action == "quit":
            self.handle.dispatch(messages.ExitEvent())

    @messages.handler
    def step(self, event: messages.StepEvent):
        now = time.monotonic()
        elapsed = now-self.last
        self.last = now
        self.accumulator = min(.1, self.accumulator+elapsed)
        simulated = 0.
        deadline = now+.012
        if not self.workshop.paused:
            while self.accumulator >= .01 and time.monotonic() < deadline:
                self.workshop.advance(.01, now=now)
                self.accumulator -= .01
                simulated += .01
            if elapsed > 0:
                self.ratio = self.ratio*.95+.05*min(2., simulated/elapsed)
        else:
            self.accumulator = 0.
        w, m, d = self.workshop, self.workshop.sim.model, self.workshop.sim.data
        copy_display_state(d, self.display_model, self.display_data)
        if now >= self.next_status:
            lifted = floor_boxes = 0
            for i in range(4):
                for j in range(4):
                    z = d.xpos[m.body(f"workshop_box_{i}_{j}").id, 2]
                    lifted += z > .762+.0292+.03
                    floor_boxes += z < .15
            joints = tuple(float(d.qpos[m.joint(s+"_"+r).qposadr[0]])
                           for s in ("left", "right") for r in ROLES)
            self.handle.send_to_viewer(WorkshopStatus(d.time, self.ratio, w.paused,
                                                       joints, w.ik_error, int(lifted), int(floor_boxes)))
            self.next_status = now+.1
        if self.feeds and now >= self.next_image:
            if self.renderer is None:
                from .rendering import create_renderer
                self.renderer = create_renderer(self.display_model, width=240, height=180, backend="filament")
            names = ("front_left", "front_right", "wrist_left", "wrist_right")
            images = self.renderer.render_many(self.display_data, names, visible_options())
            pixels = np.concatenate((np.concatenate([images[n] for n in names[:2]], axis=1),
                                     np.concatenate([images[n] for n in names[2:]], axis=1)), axis=0)
            self.handle.send_to_viewer(FeedImage(pixels))
            self.next_image = now+.2
        time.sleep(.001)

    def close_renderer(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    @messages.handler
    def close(self, event: messages.ExitEvent):
        self.workshop.sim.set_base()
        self.close_renderer()


ACCENT = (.24, .82, .89, 1)
MUTED = (.58, .67, .74, 1)


class WorkshopPanel(plugin.ViewerPlugin):
    def __init__(self):
        self.side = "left"
        self.joint = 2  # elbow is a useful first adjustment from the parked pose
        self.status = None
        self.camera_mode = 0
        self.follow = False
        self.feeds = False
        self.help = True
        self.pending_image = None
        self.texture = 0

    @messages.handler(priority=0)
    def model_loaded(self, event: messages.PostModelEvent):
        self.viewer.vis_options = visible_options()
        self.set_camera(0)

    @messages.handler
    def status_changed(self, event: WorkshopStatus):
        self.status = event

    @messages.handler
    def image_changed(self, event: FeedImage):
        self.pending_image = event.pixels

    def send(self, action):
        self.viewer.send_to_sim(WorkshopAction(action))

    def set_camera(self, mode):
        self.camera_mode = mode
        cam = self.viewer.camera
        if mode in (0, 1):
            source = overview_camera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = source.lookat
            cam.distance, cam.azimuth, cam.elevation = source.distance, source.azimuth, source.elevation
            if mode == 1:
                cam.lookat[:] = [.25, 0, .66]
                cam.distance, cam.azimuth, cam.elevation = 1.85, 135, -22
        else:
            cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            name = "front_left" if mode == 2 else "wrist_"+self.side
            cam.fixedcamid = self.viewer.model.camera(name).id

    def swap_arm(self):
        self.side = "right" if self.side == "left" else "left"
        if self.camera_mode == 3:
            self.set_camera(3)

    def toggle_feeds(self):
        self.feeds = not self.feeds
        self.send("feeds")

    @messages.handler
    def update(self, event: messages.UpdateEvent):
        key = imgui.Key
        io = imgui.GetIO()
        pressed = lambda k: imgui.IsKeyPressed(k, False)
        down = imgui.IsKeyDown
        # ImGui clears held keys when the native window loses focus. A separate
        # simulation-side heartbeat timeout also stops stale drive commands.
        controls = Controls(side=self.side, joint=self.joint)
        if not io.WantCaptureKeyboard and not (io.KeyCtrl or io.KeyAlt or io.KeySuper):
            if pressed(key.Tab):
                self.swap_arm()
            for j in range(6):
                if pressed(getattr(key, "N"+str(j+1))):
                    self.joint = j
            for k, action in ((key.P, "pause"), (key.Backspace, "reset")):
                if pressed(k):
                    self.send(action)
            if pressed(key.Space):
                self.send("hold")
            if pressed(key.C):
                self.set_camera((self.camera_mode+1) % 4)
            if pressed(key.V):
                self.follow = not self.follow
            if pressed(key.B):
                self.toggle_feeds()
            if pressed(key.H):
                self.help = not self.help
            if pressed(key.Escape):
                self.send("quit")
            if not down(key.Space):
                axis = lambda a, b: float(down(a))-float(down(b))
                controls = Controls(side=self.side, joint=self.joint,
                    drive=(axis(key.W, key.S), axis(key.A, key.D), axis(key.Q, key.E)),
                    hand=(axis(key.UpArrow, key.DownArrow), axis(key.LeftArrow, key.RightArrow), axis(key.R, key.F)),
                    wrist=(axis(key.J, key.L), axis(key.U, key.O)),
                    grip=axis(key.X, key.Z), elevator=axis(key.T, key.G),
                    jog=axis(key.RightBracket, key.LeftBracket), precision=bool(io.KeyShift))
        self.viewer.send_to_sim(ManualInput(controls))
        if self.follow and self.camera_mode in (0, 1):
            base = self.viewer.data.xpos[self.viewer.model.body("base_link").id]
            self.viewer.camera.lookat[:] = base + [0, 0, .45]
        if self.pending_image is not None:
            pixels, self.pending_image = self.pending_image, None
            h, w = pixels.shape[:2]
            self.texture = self.viewer.upload_image(self.texture, pixels.tobytes(), w, h, 3)

    @messages.handler
    def build_gui(self, event: messages.BuildGuiEvent):
        io = imgui.GetIO()
        imgui.PushStyleVar(imgui.StyleVar.WindowRounding, 12.)
        imgui.PushStyleVar(imgui.StyleVar.FrameRounding, 6.)
        imgui.PushStyleVar(imgui.StyleVar.WindowPadding, imgui.Vec2(18, 16))
        imgui.PushStyleColor(imgui.Col.WindowBg, imgui.Vec4(.045, .065, .085, .96))
        imgui.PushStyleColor(imgui.Col.Button, imgui.Vec4(.12, .22, .27, 1))
        imgui.PushStyleColor(imgui.Col.ButtonHovered, imgui.Vec4(.16, .36, .42, 1))
        imgui.SetNextWindowPos(imgui.Vec2(18, 18), imgui.Cond.Always)
        imgui.SetNextWindowSize(imgui.Vec2(320, max(260., io.DisplaySize.y-36)), imgui.Cond.Always)
        flags = sum(int(flag) for flag in (imgui.WindowFlags.NoTitleBar, imgui.WindowFlags.NoMove,
                                          imgui.WindowFlags.NoResize, imgui.WindowFlags.NoSavedSettings))
        visible, _ = imgui.Begin("Workshop controls", flags=flags)
        if visible:
            imgui.TextColored(imgui.Vec4(*ACCENT), "S O U R C C E Y")
            imgui.TextUnformatted("THE BOX WORKSHOP")
            imgui.TextColored(imgui.Vec4(*MUTED), "4 stations  /  16 physical boxes")
            imgui.Separator()
            status = self.status
            label = "PAUSED" if status and status.paused else "LIVE SIMULATION"
            imgui.TextColored(imgui.Vec4(.54, .86, .66, 1), label)
            if status:
                imgui.TextUnformatted(f"{status.sim_time:6.1f}s  |  physics {status.realtime:.2f}x real time")
                imgui.TextUnformatted(f"Lifted: {status.lifted}     On floor: {status.floor_boxes}")
            imgui.Separator()
            imgui.TextUnformatted("ACTIVE ARM")
            if imgui.Button("<  " + self.side.upper() + " ARM  >", imgui.Vec2(280, 32)):
                self.swap_arm()
            imgui.TextColored(imgui.Vec4(*MUTED), "Tab switches left / right")
            changed, self.joint = imgui.Combo("Joint", self.joint,
                [f"{i+1}. {r.replace('_', ' ')}" for i, r in enumerate(ROLES)])
            if status:
                offset = 0 if self.side == "left" else 6
                degrees = np.rad2deg(status.joints[offset+self.joint])
                imgui.TextColored(imgui.Vec4(*ACCENT), f"{degrees:+.1f} deg   |   [ / ] to jog")
            imgui.Separator()
            if imgui.Button("Hold  /  Space", imgui.Vec2(134, 30)):
                self.send("hold")
            imgui.SameLine()
            if imgui.Button("Pause  /  P", imgui.Vec2(134, 30)):
                self.send("pause")
            if imgui.Button("Reset workshop  /  Backspace", imgui.Vec2(280, 30)):
                self.send("reset")
            imgui.Separator()
            _, camera = imgui.Combo("View", self.camera_mode,
                ["Whole workshop", "Close workbench", "Left eye", "Active hand"])
            if camera != self.camera_mode:
                self.set_camera(camera)
            _, self.follow = imgui.Checkbox("Follow the robot  /  V", self.follow)
            changed, value = imgui.Checkbox("Four onboard cameras  /  B", self.feeds)
            if changed:
                self.toggle_feeds()
            _, self.help = imgui.Checkbox("Keyboard guide  /  H", self.help)
            if self.help:
                imgui.Separator()
                for title, lines in (
                    ("DRIVE", ("W / S     Forward / back", "A / D     Strafe left / right", "Q / E     Turn left / right")),
                    ("ACTIVE HAND", ("Arrows    Forward / sideways", "R / F     Raise / lower hand", "J / L     Bend wrist", "U / O     Rotate wrist", "X / Z     Open / close gripper", "T / G     Raise / lower elevator")),
                    ("PRECISE JOINTS", ("1..6      Select a joint", "[ / ]     Decrease / increase", "Shift     Quarter-speed control"))):
                    imgui.TextColored(imgui.Vec4(*ACCENT), title)
                    for line in lines:
                        imgui.TextUnformatted(line)
                imgui.TextWrapped("Tip: bend the elbow first (3 then [). Use the close view or hand camera to line up a grasp.")
            imgui.Separator()
            imgui.TextColored(imgui.Vec4(*MUTED), "Mouse drag: orbit  |  Wheel: zoom")
            imgui.TextColored(imgui.Vec4(*MUTED), "Simulation only  |  Esc exits")
        imgui.End()
        if self.feeds:
            imgui.SetNextWindowPos(imgui.Vec2(max(350., io.DisplaySize.x-506), 18), imgui.Cond.FirstUseEver)
            imgui.SetNextWindowSize(imgui.Vec2(490, 420), imgui.Cond.FirstUseEver)
            visible, _ = imgui.Begin("Eyes + hands")
            if visible:
                imgui.TextUnformatted("Left eye / Right eye     |     Left hand / Right hand")
                if self.texture:
                    imgui.Image(self.texture, imgui.Vec2(456, 342))
            imgui.End()
        imgui.PopStyleColor(3)
        imgui.PopStyleVar(3)
