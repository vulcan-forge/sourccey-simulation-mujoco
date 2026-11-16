"""Explicit classic/Filament rendering, including synchronized multi-camera batches."""
import os

import mujoco
import numpy as np


def backend_name(backend=None):
    value = backend or os.environ.get('SOURCCEY_RENDERER', 'classic')
    if value not in ('classic', 'filament'):
        raise ValueError('Renderer must be classic or filament')
    return value


def provenance(backend=None):
    return dict(mujoco_version=mujoco.__version__, renderer=backend_name(backend),
                graphics_api=os.environ.get('SOURCCEY_FILAMENT_API', 'opengl') if backend_name(backend) == 'filament' else 'opengl')


def create_renderer(model, height=240, width=320, *, backend=None):
    if backend_name(backend) == 'classic':
        return mujoco.Renderer(model, height=height, width=width)
    return FilamentRenderer(model, height=height, width=width)


class FilamentRenderer:
    """Classic-style adapter; render_many submits all cameras before one readback wait.

    Own this object on one thread. Returned images are copies, safe for encoders.
    Backend failures propagate: a dataset must never silently change renderer.
    """

    def __init__(self, model, height=240, width=320):
        from mujoco import mjrf
        from mujoco.rendering.filament.renderer import Renderer
        self.model = model
        self.width, self.height = int(width), int(height)
        if self.width <= 0 or self.height <= 0:
            raise ValueError('Image dimensions must be positive')
        api = os.environ.get('SOURCCEY_FILAMENT_API', 'opengl')
        apis = dict(default=mjrf.GraphicsApi.GRAPHICS_API_DEFAULT,
                    opengl=mjrf.GraphicsApi.GRAPHICS_API_OPENGL,
                    vulkan=mjrf.GraphicsApi.GRAPHICS_API_VULKAN)
        if api not in apis:
            raise ValueError('SOURCCEY_FILAMENT_API must be default, opengl or vulkan')
        # Filament's OpenGL backend needs a current native context on Windows.
        self.gl_context = None
        if api in ('default', 'opengl'):
            self.gl_context = mujoco.GLContext(self.width, self.height)
            self.gl_context.make_current()
        self.context = mjrf.Context(mjrf.ContextConfig(graphics_api=apis[api]))
        self.renderer = Renderer(self.context)
        self.scene = self.renderer.scene('robot')
        self.scene.configure_from_model(model)
        self.resources = mjrf.ModelResources(self.context, model)
        self.lights = mjrf.ModelLights(self.scene, self.resources)
        self.renderables = mjrf.ModelRenderables(self.scene, self.resources)
        self.buffers = {}
        self.data = None
        self.camera = -1
        self.options = mujoco.MjvOption()
        self.closed = False

    def update_scene(self, data, camera=-1, scene_option=None):
        if self.closed:
            raise RuntimeError('Renderer is closed')
        self.data, self.camera = data, camera
        self.options = scene_option or mujoco.MjvOption()

    def _camera(self, value):
        if isinstance(value, mujoco.MjvCamera):
            return value
        cam = mujoco.MjvCamera()
        cid = self.model.camera(value).id if isinstance(value, str) else int(value)
        if cid >= 0:
            if cid >= self.model.ncam:
                raise ValueError(f'Unknown camera: {value}')
            cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            cam.fixedcamid = cid
        else:
            mujoco.mjv_defaultFreeCamera(self.model, cam)
        return cam

    def render_many(self, data, cameras, scene_option=None):
        from mujoco.rendering.filament.renderer import View
        if self.closed:
            raise RuntimeError('Renderer is closed')
        if self.gl_context is not None:
            self.gl_context.make_current()
        self.renderables.set_options(scene_option or mujoco.MjvOption())
        self.lights.update(data)
        self.renderables.update(data)
        views = []
        cameras = tuple(cameras)
        if not cameras:
            return {}
        # 4.0.1 supports one read request per frame: use a horizontal atlas.
        # All views still share one state update and one GPU submission/wait.
        key = str(len(cameras))
        if key not in self.buffers:
            self.buffers[key] = np.empty((self.height, self.width*len(cameras), 3), dtype=np.uint8)
            self.renderer.target(key, self.buffers[key])
        for index, name in enumerate(cameras):
            cam = mujoco.mjv_camera2GLCamera(self.model, data, self._camera(name))
            views.append(View(scene='robot', target=key, camera=cam,
                              viewport=(index/len(cameras), 0., 1/len(cameras), 1.),
                              headlight=bool(self.model.vis.headlight.active),
                              headlight_color=tuple(self.model.vis.headlight.diffuse),
                              headlight_intensity=40000. if not np.any(self.model.light_intensity > 0) else 0.))
        self.renderer.render(views)
        return {(id(name) if isinstance(name,mujoco.MjvCamera) else name):
                    self.buffers[key][:, index*self.width:(index+1)*self.width].copy()
                for index, name in enumerate(cameras)}

    def render(self):
        if self.data is None:
            raise RuntimeError('Call update_scene before render')
        key = id(self.camera) if isinstance(self.camera,mujoco.MjvCamera) else self.camera
        return self.render_many(self.data, [self.camera], self.options)[key]

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.gl_context is not None:
            self.gl_context.make_current()
        self.renderables = self.lights = self.resources = None
        self.scene = self.renderer = None
        self.buffers.clear()
        self.context = None
        if self.gl_context is not None:
            self.gl_context.free()
            self.gl_context = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
