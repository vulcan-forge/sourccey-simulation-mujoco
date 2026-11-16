"""Studio plugins for recorded experiment review; no hardware connections."""
from dataclasses import dataclass
import time

import mujoco
import numpy as np
from mujoco.studio import messages, plugin
from mujoco.studio.dear_imgui import dear_imgui as imgui

from examples.box_pickup import camera, options
from examples.box_training import SIZE, mosaic, render_camera_frames, camera_frusta
from sourccey.rendering import create_renderer


class NativePresentation(plugin.ViewerPlugin):
    """4.0.1 workaround: Python Renderer omits its direct-window target path.

    Keep the workaround local to this viewer; never patch the installed package.
    Offscreen targets and the web viewer use the upstream implementation.
    """
    @messages.handler
    def initialize(self, event: plugin.ViewerInitEvent):
        if mujoco.__version__ != '4.0.1' or self.viewer.config.gfx != 'opengl':
            return
        from mujoco import mjrf
        from mujoco.rendering.filament.renderer import Renderer
        from mujoco.studio import native_viewer_cc
        window = self.viewer._window
        # The upstream constructor also omits the native swap-chain window.
        context = mjrf.Context(mjrf.ContextConfig(
            graphics_api=mjrf.GraphicsApi.GRAPHICS_API_OPENGL,
            native_window=window.GetNativeWindowHandle()))
        renderer = Renderer(context)
        self.viewer._imgui_bridge = native_viewer_cc.ImguiBridge(context,renderer.scene('ux'))
        self.viewer._renderer = renderer
        original = renderer.render
        def render(views):
            if all(view.target for view in views):
                return original(views)
            width = int(window.GetWidth()*window.GetScale())
            height = int(window.GetHeight()*window.GetScale())
            requests = []
            for view in views:
                request = mjrf.RenderRequest(scene=renderer.scene(view.scene),
                    camera=renderer._to_mjrf_camera(view.camera))
                request.viewport.left = int(view.viewport[0]*width)
                request.viewport.bottom = int(view.viewport[1]*height)
                request.viewport.width = int(view.viewport[2]*width)
                request.viewport.height = int(view.viewport[3]*height)
                for name in ('draw_mode','post_processing','reflections','shadows','skybox','haze',
                             'headlight','headlight_color','headlight_intensity'):
                    setattr(request,name,getattr(view,name))
                requests.append(request)
            frame = renderer.context.render(requests,[])
            renderer.context.wait_for_frame(frame)
        renderer.render = render

    @messages.handler
    def close(self, event: messages.ExitEvent):
        renderer = getattr(self.viewer,'_renderer',None)
        if renderer is not None and 'render' in vars(renderer):
            # Break the local compatibility closure on the owning render thread.
            del renderer.render


@dataclass(frozen=True)
class PlaybackCommand(messages.Snapshot):
    paused: bool
    speed: float
    seek: float | None = None


@dataclass(frozen=True)
class PlaybackStatus(messages.Snapshot):
    position: float
    duration: float
    phase: str
    passed: bool


@dataclass(frozen=True)
class CameraImage(messages.Snapshot):
    pixels: np.ndarray


class RecordedExperiment(plugin.SimPlugin):
    def __init__(self, model, data, recording, report, speed=1.):
        self.model, self.data = model, data
        self.recording, self.report = recording, report
        self.speed, self.paused = speed, False
        self.position = float(recording['time'][0])
        self.last = time.perf_counter()
        self.next_camera = 0.
        self.renderer = None

    @messages.handler
    def command(self, event: PlaybackCommand):
        self.paused, self.speed = event.paused, event.speed
        if event.seek is not None:
            self.position = float(np.clip(event.seek, 0., self.recording['time'][-1]))

    @messages.handler
    def step(self, event: messages.StepEvent):
        now = time.perf_counter()
        if not self.paused:
            self.position = min(float(self.recording['time'][-1]), self.position+(now-self.last)*self.speed)
        self.last = now
        index = int(np.clip(np.searchsorted(self.recording['time'], self.position, side='right')-1,
                            0, len(self.recording['time'])-1))
        self.data.qpos[:] = self.recording['qpos'][index]
        self.data.time = self.position
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)
        self.handle.send_to_viewer(PlaybackStatus(self.position, float(self.recording['time'][-1]),
            str(self.recording['phase'][index]), bool(self.report['passed'])))
        if now >= self.next_camera:
            if self.renderer is None:
                self.renderer = create_renderer(self.model, width=SIZE[0], height=SIZE[1], backend='filament')
            pixels = mosaic(render_camera_frames(self.model,self.data,self.renderer,self.report['scene'],index),width=240)
            self.handle.send_to_viewer(CameraImage(pixels))
            self.next_camera = now+.1
        time.sleep(.008)

    @messages.handler
    def close(self, event: messages.ExitEvent):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


class ExperimentPanel(plugin.ViewerPlugin):
    paused = False
    speed = 1.
    frusta = True
    texture = 0
    pixels = None
    pending_image = None
    status = None
    scene = None

    @messages.handler(priority=0)  # After Studio App's INTERNAL camera reset.
    def model_loaded(self, event: messages.PostModelEvent):
        self.scene = None
        self.viewer.vis_options = options()
        cam = camera()
        self.viewer.camera.lookat[:] = cam.lookat
        self.viewer.camera.distance = cam.distance
        self.viewer.camera.azimuth = cam.azimuth
        self.viewer.camera.elevation = cam.elevation

    @messages.handler
    def status_changed(self, event: PlaybackStatus):
        self.status = event

    @messages.handler
    def image_changed(self, event: CameraImage):
        self.pending_image = event.pixels

    @messages.handler
    def update(self, event: messages.UpdateEvent):
        if self.pending_image is not None:
            self.pixels, self.pending_image = self.pending_image, None
            height, width = self.pixels.shape[:2]
            self.texture = self.viewer.upload_image(self.texture, self.pixels.tobytes(),width,height,3)
        self.viewer.extra_geoms.clear()
        if self.frusta and self.viewer.model.ncam >= 4:
            if self.scene is None:
                self.scene = mujoco.MjvScene(self.viewer.model, maxgeom=64)
            camera_frusta(self.scene, self.viewer.model, self.viewer.data)
            self.viewer.extra_geoms.extend(self.scene.geoms[:self.scene.ngeom])

    @messages.handler
    def build_gui(self, event: messages.BuildGuiEvent):
        imgui.SetNextWindowPos(imgui.Vec2(12,60),imgui.Cond.FirstUseEver)
        imgui.SetNextWindowSize(imgui.Vec2(430,220),imgui.Cond.FirstUseEver)
        visible, _ = imgui.Begin('Sourccey experiment')
        if visible:
            changed, self.paused = imgui.Checkbox('Pause', self.paused)
            speed_changed, self.speed = imgui.SliderFloat('Playback speed', self.speed,.1,3.)
            seek = None
            restart = imgui.Button('Restart')
            if restart:
                seek, self.paused = 0., False
            if self.status:
                moved, value = imgui.SliderFloat('Time', self.status.position,0.,self.status.duration,'%.2f s')
                if moved:
                    seek = value
                imgui.TextUnformatted(f'{self.status.phase} | Physics: {"PASS" if self.status.passed else "FAIL"}')
            if changed or speed_changed or seek is not None:
                self.viewer.send_to_sim(PlaybackCommand(self.paused,self.speed,seek))
            _, self.frusta = imgui.Checkbox('Camera frusta',self.frusta)
            imgui.TextUnformatted('Recorded simulation; camera calibration remains provisional.')
        imgui.End()
        imgui.SetNextWindowPos(imgui.Vec2(690,60),imgui.Cond.FirstUseEver)
        imgui.SetNextWindowSize(imgui.Vec2(500,590),imgui.Cond.FirstUseEver)
        visible, _ = imgui.Begin('Four onboard cameras')
        if visible and self.texture and self.pixels is not None:
            height,width = self.pixels.shape[:2]
            imgui.Image(self.texture,imgui.Vec2(width,height))
        imgui.End()
