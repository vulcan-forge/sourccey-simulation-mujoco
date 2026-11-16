"""Procedural PBR surfaces; estimated appearance, never physical calibration."""
import mujoco
import numpy as np


def add_box_materials(spec, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:128, :128]
    definitions = (
        ('cardboard', [.66,.45,.25], .93, 0., .025),
        ('wood', [.43,.28,.15], .72, 0., .035),
        ('shell', [.72,.75,.78], .55, 0., .006),
        ('gripper', [.20,.34,.43], .82, 0., .012),
        ('metal', [.25,.27,.29], .32, .8, .006))
    for name, color, roughness, metallic, noise in definitions:
        grain = .035*np.sin(xx*.6 + .6*np.sin(yy*.12)) if name == 'wood' else 0.
        values = np.asarray(color)[None,None,:] + rng.normal(0,noise,(128,128,1))
        values = values + (grain[:,:,None] if isinstance(grain,np.ndarray) else grain)
        texture = spec.add_texture(name='pbr_'+name, type=mujoco.mjtTexture.mjTEXTURE_2D,
            width=128, height=128, nchannel=3, data=np.uint8(np.clip(values,0,1)*255).ravel())
        material = spec.add_material(name='pbr_'+name, rgba=[1,1,1,1],
            roughness=roughness, metallic=metallic, texuniform=True, texrepeat=[2,2],
            specular=.18, shininess=.12)
        material.textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = texture.name
    for geom in spec.geoms:
        name = geom.name
        role = ('cardboard' if name == 'pickup_box_geom' else 'wood' if name == 'box_test_table'
                else 'metal' if name.startswith('box_test_leg_') else 'gripper' if name.startswith('visual_Gripper_')
                else 'shell' if name.startswith('visual_') and geom.rgba[0] > .4 else None)
        if role:
            geom.material = 'pbr_'+role
            geom.rgba = [1,1,1,1]


def randomize_pbr(spec, parameters):
    """Randomize reflectance without altering geometry, camera poses or dynamics."""
    rng = np.random.default_rng(np.random.SeedSequence([parameters['seed'], 4001]))
    for material in spec.materials:
        if material.name.startswith('episode_material_'):
            material.roughness = float(rng.uniform(.60,.98))
            material.metallic = 0.
    # Keep all intensity values at zero: ModelLights supplies the fallback IBL.
    # Introducing one photometric light without an environment map darkens the scene.
    for light in spec.lights:
        light.bulbradius = float(rng.uniform(.015,.08))
