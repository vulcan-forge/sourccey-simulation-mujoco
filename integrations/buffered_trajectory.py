"""Portable, hardware-independent buffered joint trajectory and local executor.

This module is also installed beside the robot Host. It needs only Python's
standard library; uploading constructs a trajectory but never starts motion.
"""
from bisect import bisect_right
import hashlib
import json
import math

PROTOCOL = 1
PORT = 5557
ROLES = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')
KEYS = tuple(f'{side}_{role}.pos' for side in ('left', 'right') for role in ROLES)
MAX_POINTS = 10000
MAX_BYTES = 2 * 1024 * 1024
LEASE_S = 5.
LOCAL_FEEDBACK_TIMEOUT_S = .5
MAX_STEP = 1.


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Expected a finite numeric value')
    return float(value)


def canonical(payload):
    return json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def checksum(payload):
    return hashlib.sha256(canonical(payload)).hexdigest()


def validate_pose(pose):
    if set(pose) != set(KEYS):
        raise ValueError('Expected exactly twelve arm/gripper joint positions')
    result = {key: number(pose[key]) for key in KEYS}
    for key, value in result.items():
        if not (0 if key.endswith('gripper.pos') else -100) <= value <= 100:
            raise ValueError(f'{key} outside normalized hardware range')
    return result


class Trajectory:
    """Componentwise PCHIP, matching the existing SciPy replay interpolation."""
    def __init__(self, payload, limits=None):
        if len(canonical(payload)) > MAX_BYTES:
            raise ValueError('Trajectory exceeds upload size limit')
        if payload.get('version') != PROTOCOL or payload.get('units') != 'normalized':
            raise ValueError('Trajectory protocol/units mismatch')
        if payload.get('keys') != list(KEYS):
            raise ValueError('Trajectory joint order mismatch')
        self.times = tuple(number(t) for t in payload['times'])
        rows = payload['positions']
        if not 2 <= len(self.times) <= MAX_POINTS or len(rows) != len(self.times):
            raise ValueError('Invalid trajectory point count')
        if self.times[0] != 0 or not 0 < self.times[-1] <= 600:
            raise ValueError('Trajectory must start at zero and last at most 600 s')
        h = [b - a for a, b in zip(self.times, self.times[1:])]
        if any(v < 1e-5 for v in h):
            raise ValueError('Trajectory timestamps must strictly increase')
        self.rows = []
        for row in rows:
            if len(row) != len(KEYS):
                raise ValueError('Invalid trajectory row length')
            pose = validate_pose(dict(zip(KEYS, row)))
            for key, bounds in (limits or {}).items():
                if not bounds[0] <= pose[key] <= bounds[1]:
                    raise ValueError(f'Trajectory exceeds Host guard at {key}')
            self.rows.append(tuple(pose[k] for k in KEYS))
        self.elevator = number(payload['elevator'])
        self.tolerance = number(payload['tracking_tolerance'])
        self.stall_timeout = number(payload['stall_timeout'])
        if not 0 < self.tolerance < 10 or not 0 < self.stall_timeout <= 120:
            raise ValueError('Invalid tracking tolerance or stall timeout')
        self.slopes = [[0.] * len(KEYS) for _ in self.times]
        for j in range(len(KEYS)):
            d = [(b[j] - a[j]) / dt for a, b, dt in zip(self.rows, self.rows[1:], h)]
            if len(d) == 1:
                self.slopes[0][j] = self.slopes[-1][j] = d[0]
                continue
            def edge(h0, h1, d0, d1):
                m = ((2*h0 + h1)*d0 - h0*d1) / (h0 + h1)
                if m*d0 <= 0:
                    return 0.
                return 3*d0 if d0*d1 <= 0 and abs(m) > abs(3*d0) else m
            self.slopes[0][j] = edge(h[0], h[1], d[0], d[1])
            self.slopes[-1][j] = edge(h[-1], h[-2], d[-1], d[-2])
            for i in range(1, len(self.times)-1):
                if d[i-1]*d[i] > 0:
                    w1, w2 = 2*h[i] + h[i-1], h[i] + 2*h[i-1]
                    self.slopes[i][j] = (w1+w2) / (w1/d[i-1] + w2/d[i])

    def at(self, elapsed):
        t = min(self.times[-1], max(0., elapsed))
        i = min(len(self.times)-2, max(0, bisect_right(self.times, t)-1))
        h = self.times[i+1] - self.times[i]
        s = (t-self.times[i]) / h
        result = {key: ((2*s**3-3*s*s+1)*self.rows[i][j]
                      + (s**3-2*s*s+s)*h*self.slopes[i][j]
                      + (-2*s**3+3*s*s)*self.rows[i+1][j]
                      + (s**3-s*s)*h*self.slopes[i+1][j])
                for j, key in enumerate(KEYS)}
        # PCHIP is bounded by each component's adjacent knots. Remove only
        # floating-point roundoff at normalized endpoints, never extrapolate.
        return {key: min(max(value, min(self.rows[i][j], self.rows[i+1][j])),
                         max(self.rows[i][j], self.rows[i+1][j]))
                for j, (key, value) in enumerate(result.items())}

    def advance(self, elapsed, last, observed, dt):
        horizon = min(self.times[-1], elapsed + max(0., dt))
        step_limit = min(MAX_STEP, self.tolerance/2)
        def allowed(target):
            return (max(abs(target[k]-observed[k]) for k in KEYS) <= self.tolerance
                    and max(abs(target[k]-last[k]) for k in KEYS) <= step_limit)
        if not allowed(last):
            return elapsed, last, True
        lower, target = elapsed, last
        knots = self.times[bisect_right(self.times, elapsed):bisect_right(self.times, horizon)]
        for upper in (*knots, horizon):
            candidate = self.at(upper)
            if allowed(candidate):
                lower, target = upper, candidate
                continue
            for _ in range(24):
                middle = (lower+upper)/2
                candidate = self.at(middle)
                if allowed(candidate):
                    lower, target = middle, candidate
                else:
                    upper = middle
            return lower, target, True
        return horizon, target, False


def tracking(observed, target):
    joint = max(KEYS, key=lambda k: abs(observed[k]-target[k]))
    return dict(joint=joint, observed=observed[joint], commanded=target[joint],
                error_units=abs(observed[joint]-target[joint]))


class Executor:
    """Local feedback, independent of the PC's status/rendering cadence."""
    def __init__(self, trajectory, period=1/30):
        self.path, self.period = trajectory, period
        self.state, self.error = 'uploaded', None
        self.elapsed = 0.
        self.target = trajectory.at(0.)
        self.started = self.previous = self.heartbeat = self.progress_at = None
        self.progress_target = self.target.copy()
        self.settled = self.final_deadline = None
        self.finished_at = None
        self.observed = None
        self.observed_at = None
        self.last_tracking = None
        self.limited, self.limit_reason, self.command_step = False, None, 0.

    def validate_feedback(self, observation):
        current = validate_pose(observation['positions'])
        if abs(number(observation['elevator'])-self.path.elevator) > 2:
            raise ValueError('Elevator moved during buffered replay')
        velocities = observation['base_velocity']
        if len(velocities) != 4 or any(abs(number(v)) > .02 for v in velocities):
            raise ValueError('Base is moving during buffered replay')
        return current

    def start(self, now, observation):
        if self.state != 'uploaded':
            return  # A retried start can never restart completed/cancelled motion.
        current = self.validate_feedback(observation)
        error = tracking(current, self.target)
        if error['error_units'] > self.path.tolerance:
            raise ValueError(f"Starting pose differs at {error['joint']}: {error['error_units']:.3f} units")
        self.started = self.previous = self.heartbeat = self.progress_at = now
        self.observed, self.observed_at = observation, now
        self.state = 'running'

    def abort(self, reason, state='aborted', now=None):
        if self.state in ('uploaded', 'running'):
            self.state, self.error = state, str(reason)
            self.finished_at = self.previous if now is None else now

    def tick(self, now, observation, send):
        if self.state != 'running':
            return
        try:
            if now-self.heartbeat >= LEASE_S:
                raise TimeoutError('Client lease expired; last target held')
            if now-self.previous > LOCAL_FEEDBACK_TIMEOUT_S:
                raise TimeoutError('Local joint feedback exceeded 500 ms; last target held')
            current = self.validate_feedback(observation)
            error = tracking(current, self.target)
            if error['error_units'] > 10:
                raise ValueError(f"Tracking error exceeds 10 units at {error['joint']}")
            self.observed, self.observed_at = observation, now
            dt, self.previous = min(max(now-self.previous, 0), self.period), now
            elapsed, target, limited = self.path.advance(self.elapsed, self.target, current, dt)
            error = tracking(current, target)
            if not limited or max(abs(target[k]-self.progress_target[k]) for k in KEYS) >= min(.1, self.path.tolerance/10):
                self.progress_at, self.progress_target = now, target.copy()
            if limited and now-self.progress_at > self.path.stall_timeout:
                raise TimeoutError(f"No path progress for {self.path.stall_timeout:g} s at {error['joint']}")
            step = max(abs(target[k]-self.target[k]) for k in KEYS)
            send(target)  # Commit path progress only after a successful motor write.
            self.elapsed, self.target = elapsed, target
            self.last_tracking = {**error, 'tolerance_units': self.path.tolerance,
                                  'within_tolerance': error['error_units'] <= self.path.tolerance}
            self.limited, self.command_step = limited, step
            self.limit_reason = ('tracking' if error['error_units'] >= self.path.tolerance-1e-5
                                 else 'command-step') if limited else None
            if elapsed >= self.path.times[-1]:
                self.final_deadline = self.final_deadline or now + max(8., self.path.stall_timeout)
                self.settled = (now if self.settled is None else self.settled) if error['error_units'] <= self.path.tolerance else None
                if self.settled is not None and now-self.settled >= 1:
                    self.state = 'complete'
                    self.finished_at = now
                elif now > self.final_deadline:
                    raise TimeoutError('Final buffered pose did not settle')
        except Exception as exc:
            self.abort(exc, now=now)

    def status(self, now):
        return dict(state=self.state, error=self.error, trajectory_s=self.elapsed,
                    nominal_duration_s=self.path.times[-1],
                    wall_elapsed_s=0. if self.started is None else (self.finished_at if self.finished_at is not None else now)-self.started,
                    commanded=self.target, observed=self.observed,
                    observation_age_s=None if self.observed_at is None else now-self.observed_at,
                    tracking=self.last_tracking, path_limited=self.limited,
                    limit_reason=self.limit_reason, command_step_units=self.command_step,
                    waiting_for_tracking=self.limit_reason == 'tracking')
