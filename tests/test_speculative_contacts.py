import mujoco
import pytest


def test_speculative_contact_stops_a_fast_body_crossing_a_thin_obstacle():
    if not hasattr(mujoco.mjtEnableBit,'mjENBL_SPECULATIVE'):
        pytest.skip('MuJoCo 4 speculative contacts')
    def run(enabled):
        model = mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".02" gravity="0 0 0"/>
          <worldbody><geom name="obstacle" type="box" size="1 1 .01"/>
          <body pos="0 0 .5"><freejoint/><geom type="sphere" size=".03" mass="1"/>
          </body></worldbody></mujoco>''')
        if enabled:
            model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_SPECULATIVE
        data = mujoco.MjData(model)
        data.qvel[2] = -50
        mujoco.mj_step(model,data)
        return data.qpos[2]
    assert run(False) < -.4
    assert run(True) > 0.
