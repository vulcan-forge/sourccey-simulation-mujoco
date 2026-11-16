"""Offline, bounded MuJoCo sysid with held-out validation. Never connects to hardware."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_model(path):
    path = Path(path)
    return (mujoco.MjModel.from_binary_path(str(path)) if path.suffix == '.mjb'
            else mujoco.MjModel.from_xml_path(str(path)))


def load_measurements(path, model):
    """SI units in MuJoCo model order. Samples observe state BEFORE upcoming ctrl."""
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key].copy() for key in archive.files}
    required = ('time', 'qpos', 'qvel', 'ctrl', 'observed_joint_names')
    if any(key not in result for key in required):
        raise ValueError(f'Measurements require {required}')
    times = result['time']
    if times.ndim != 1 or len(times) < 10 or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError('At least ten increasing timestamps starting at zero are required')
    for key, size in (('qpos',model.nq), ('qvel',model.nv), ('ctrl',model.nu)):
        if result[key].shape != (len(times),size) or not np.isfinite(result[key]).all():
            raise ValueError(f'{key} must be finite, shape {(len(times),size)} in model order')
    if not np.isfinite(times).all():
        raise ValueError('Timestamps must be finite')
    names = result['observed_joint_names'].tolist()
    if not isinstance(names,list) or not names or len(set(names)) != len(names):
        raise ValueError('observed_joint_names must be a nonempty unique string array')
    for name in names:
        if model.joint(name).type[0] not in (mujoco.mjtJoint.mjJNT_HINGE,mujoco.mjtJoint.mjJNT_SLIDE):
            raise ValueError('Fit observations must name scalar hinge/slide joints')
    if np.min(np.diff(times)) < model.opt.timestep*.99:
        raise ValueError('Observation spacing must be at least one physics step')
    return result


def apply_parameters(model, definitions, values):
    for definition,value in zip(definitions,values,strict=True):
        kind, target = definition['kind'],definition['target']
        if kind == 'joint_damping':
            model.dof_damping[model.joint(target).dofadr[0]] = value
        elif kind == 'actuator_kp':
            actuator = model.actuator(target).id
            model.actuator_gainprm[actuator,0] = value
            model.actuator_biasprm[actuator,1] = -value
        elif kind == 'actuator_kv':
            model.actuator_biasprm[model.actuator(target).id,2] = -value
        elif kind == 'body_mass':
            body = model.body(target).id
            model.body_inertia[body] *= value/model.body_mass[body]
            model.body_mass[body] = value
        elif kind == 'sliding_friction':
            geom = model.geom(target).id
            model.geom_friction[geom,0] = value
            for pair in range(model.npair):
                if geom in (model.pair_geom1[pair],model.pair_geom2[pair]):
                    model.pair_friction[pair,:2] = value
        else:
            raise ValueError(f'Unsupported parameter kind: {kind}')


def predict(baseline, measurements, definitions=(), values=(), *, gravity_feedforward=False):
    """Use the same split-step arm feed-forward as the box teacher when requested.

    No equivalent mecanum traction is applied. Use supported/fixed-base recordings
    with a model containing the matching fixture constraints for arm identification.
    """
    model = copy.copy(baseline)
    apply_parameters(model,definitions,values)
    data = mujoco.MjData(model)
    mujoco.mj_setConst(model,data)
    data.qpos[:] = measurements['qpos'][0]
    data.qvel[:] = measurements['qvel'][0]
    names = measurements['observed_joint_names'].tolist()
    qp = [model.joint(name).qposadr[0] for name in names]
    result = [data.qpos[qp].copy()]
    targets = measurements['time']
    source_index = 0
    feedforward = []
    if gravity_feedforward:
        for aid in range(model.nu):
            joint = model.actuator_trnid[aid,0]
            if (model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT
                    and model.jnt_type[joint] in (mujoco.mjtJoint.mjJNT_HINGE,mujoco.mjtJoint.mjJNT_SLIDE)
                    and not model.actuator(aid).name.endswith('_wheel')):
                feedforward.append((aid,model.jnt_dofadr[joint]))
    for target in targets[1:]:
        while data.time < target-model.opt.timestep/2:
            while source_index+1 < len(targets) and targets[source_index+1] <= data.time+1e-10:
                source_index += 1
            data.ctrl[:] = measurements['ctrl'][source_index]
            if gravity_feedforward:
                mujoco.mj_step1(model,data)
                for aid,dof in feedforward:
                    model.actuator_biasprm[aid,0] = data.qfrc_bias[dof]
                mujoco.mj_step2(model,data)
            else:
                mujoco.mj_step(model,data)
            if not np.isfinite(data.qpos).all() or any(w.number for w in data.warning):
                raise ValueError('Unhealthy dynamics during identification')
        if abs(data.time-target) > model.opt.timestep*.01:
            raise ValueError('Measurement times must align with the physics clock')
        result.append(data.qpos[qp].copy())
    return np.asarray(result)


def fit(model_path, train_path, validation_path, config_path, output):
    from mujoco import sysid
    output = Path(output)
    if output.exists():
        raise FileExistsError('Choose a new output directory to preserve prior fits')
    if Path(train_path).resolve() == Path(validation_path).resolve() or digest(train_path) == digest(validation_path):
        raise ValueError('Training and validation must be separate recordings')
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    if not isinstance(config.get('gravity_feedforward'),bool):
        raise ValueError('Explicitly set gravity_feedforward true or false for the recording controller')
    definitions = config['parameters']
    if not definitions or len({d['name'] for d in definitions}) != len(definitions):
        raise ValueError('Parameter names must be unique and nonempty')
    model = load_model(model_path)
    train = load_measurements(train_path,model)
    validation = load_measurements(validation_path,model)
    if train['observed_joint_names'].tolist() != validation['observed_joint_names'].tolist():
        raise ValueError('Train and validation observations must name the same joints')
    params = sysid.ParameterDict()
    for definition in definitions:
        lo, initial, hi = (float(definition[k]) for k in ('min','initial','max'))
        if not np.isfinite([lo,initial,hi]).all() or not 0 <= lo < initial < hi:
            raise ValueError('Require finite 0 <= min < initial < max')
        params.add(sysid.Parameter(definition['name'],nominal=initial,min_value=lo,max_value=hi))
    qp = [model.joint(name).qposadr[0] for name in train['observed_joint_names']]
    observed = train['qpos'][:,qp]
    if np.max(np.ptp(observed,axis=0)) < .02:
        raise ValueError('Training recording lacks useful joint excitation')
    kwargs = dict(gravity_feedforward=config['gravity_feedforward'])
    initial = params.as_vector()
    def residual(values, parameter_dict):
        prediction = predict(model,train,definitions,values,**kwargs)
        return [(prediction-observed).ravel()], [], []
    fitted, optimization = sysid.optimize(params,residual,optimizer='scipy',
        max_iters=config.get('max_iterations',60),x_scale='jac',verbose=False)
    values = fitted.as_vector()
    before = predict(model,validation,definitions,initial,**kwargs)
    after = predict(model,validation,definitions,values,**kwargs)
    actual = validation['qpos'][:,qp]
    rmse_before = np.sqrt(np.mean((before-actual)**2,axis=0))
    rmse_after = np.sqrt(np.mean((after-actual)**2,axis=0))
    rank = int(np.linalg.matrix_rank(optimization.jac))
    report = dict(mujoco_version=mujoco.__version__,model_sha256=digest(model_path),
        train_sha256=digest(train_path),validation_sha256=digest(validation_path),
        observed_joint_names=train['observed_joint_names'].tolist(),
        validation_rmse_before=rmse_before.tolist(),validation_rmse_after=rmse_after.tolist(),
        jacobian_rank=rank,parameter_count=len(values),optimizer_success=bool(optimization.success),
        validated=bool(optimization.success and rank == len(values) and np.mean(rmse_after) < np.mean(rmse_before)
                       and np.all(rmse_after <= rmse_before*1.05+1e-5)),
        automatically_applied=False,parameters=[dict(d,value=float(v)) for d,v in zip(definitions,values)],
        gravity_feedforward=config['gravity_feedforward'],
        note='Observed hinge RMSE is radians, slide RMSE meters. A fit is a candidate, not proof of physical calibration.')
    output.mkdir(parents=True)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    fitted_model = copy.copy(model)
    apply_parameters(fitted_model,definitions,values)
    mujoco.mj_setConst(fitted_model,mujoco.MjData(fitted_model))
    mujoco.mj_saveModel(fitted_model,str(output/'candidate.mjb'),None)
    np.savez_compressed(output/'validation.npz',time=validation['time'],observed=actual,before=before,after=after)
    import plotly.graph_objects as go
    figure = go.Figure()
    for index,name in enumerate(report['observed_joint_names']):
        for label,series in (('measured',actual),('before',before),('fitted',after)):
            figure.add_scatter(x=validation['time'],y=series[:,index],name=f'{name}: {label}')
    figure.update_layout(title='Held-out physics identification',xaxis_title='Time (s)',yaxis_title='Joint position (rad / m)')
    figure.write_html(output/'validation.html',include_plotlyjs=True)
    return report


def audit_trace(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    replay = [r for r in rows if r.get('phase') == 'replay' and 'observed' in r and 'commanded' in r]
    times = np.array([r['wall_elapsed_s'] for r in replay if 'wall_elapsed_s' in r])
    gaps = np.diff(times)
    return dict(source=str(path),replay_samples=len(replay),
        median_status_interval_s=float(np.median(gaps)) if len(gaps) else None,
        suitable_for_dynamic_fitting=False,
        reason='Status logs omit high-rate applied-command and observation acquisition timestamps. Capture aligned model-order measurements before fitting dynamics.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path)
    parser.add_argument('--train',type=Path)
    parser.add_argument('--validation',type=Path)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--audit-trace',type=Path)
    args = parser.parse_args()
    if args.audit_trace:
        report = audit_trace(args.audit_trace)
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    else:
        if not all((args.model,args.train,args.validation,args.config,args.output)):
            parser.error('Fit requires --model, --train, --validation, --config and --output')
        report = fit(args.model,args.train,args.validation,args.config,args.output)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
