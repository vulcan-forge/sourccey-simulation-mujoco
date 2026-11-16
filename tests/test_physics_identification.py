import json

import mujoco
import numpy as np
import pytest

from integrations.identify_physics import fit,load_measurements,predict


def measurements(model,phase=0.):
    times = np.arange(401)*model.opt.timestep
    data = dict(time=times,qpos=np.zeros((len(times),model.nq)),qvel=np.zeros((len(times),model.nv)),
        ctrl=(.4*np.sin(times*11+phase))[:,None],observed_joint_names=np.array(['hinge']))
    data['qpos'][:] = predict(model,data)
    return data


def test_fit_recovers_physics_and_improves_separate_recording(tmp_path):
    if int(mujoco.__version__.split('.')[0]) < 4:
        pytest.skip('MuJoCo 4 sysid toolbox')
    model_file = tmp_path/'model.xml'
    model_file.write_text('''<mujoco><option timestep=".002" gravity="0 0 0"/>
      <worldbody><body><joint name="hinge" damping=".2"/>
      <geom type="capsule" fromto="0 0 0 0 0 .3" size=".03" mass="1"/>
      </body></worldbody><actuator><position name="servo" joint="hinge" kp="8" kv=".3"/></actuator></mujoco>''')
    baseline = mujoco.MjModel.from_xml_path(str(model_file))
    baseline.dof_damping[:] = 1.7
    train,validation = measurements(baseline),measurements(baseline,.7)
    np.savez(tmp_path/'train.npz',**train)
    np.savez(tmp_path/'validation.npz',**validation)
    config = dict(gravity_feedforward=False,parameters=[dict(name='damping',kind='joint_damping',
        target='hinge',initial=.2,min=.01,max=4.)])
    (tmp_path/'config.json').write_text(json.dumps(config))
    report = fit(model_file,tmp_path/'train.npz',tmp_path/'validation.npz',tmp_path/'config.json',tmp_path/'fit')
    assert report['validated'],report
    assert report['parameters'][0]['value'] == pytest.approx(1.7,abs=.02)
    assert report['validation_rmse_after'][0] < report['validation_rmse_before'][0]*.01
    assert not report['automatically_applied']
    assert (tmp_path/'fit/validation.html').exists()
    assert mujoco.MjModel.from_binary_path(str(tmp_path/'fit/candidate.mjb')).dof_damping[0] == pytest.approx(1.7,abs=.02)
    with pytest.raises(ValueError,match='separate'):
        fit(model_file,tmp_path/'train.npz',tmp_path/'train.npz',tmp_path/'config.json',tmp_path/'other')


def test_rejects_misaligned_measurements(tmp_path):
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body><joint name="hinge"/>
      <geom size=".1"/></body></worldbody><actuator><motor joint="hinge"/></actuator></mujoco>''')
    data = measurements(model)
    data['time'][20] += model.opt.timestep*.3
    np.savez(tmp_path/'bad.npz',**data)
    with pytest.raises(ValueError,match='physics step'):
        load_measurements(tmp_path/'bad.npz',model)
    data['time'] = np.arange(401)*model.opt.timestep*2
    data['time'][20] += model.opt.timestep*.3
    np.savez(tmp_path/'bad.npz',**data)
    loaded = load_measurements(tmp_path/'bad.npz',model)
    with pytest.raises(ValueError,match='align'):
        predict(model,loaded)
