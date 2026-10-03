"""Check the virtual optical point against the actual front-panel CAD slice."""
import numpy as np

from sourccey.build import CAD_ROTATION, SOURCE, vertices
from sourccey.lidar import ANGLES, CONFIG


def panel_cross_section(z):
    path = SOURCE.parent / 'meshes/0029_Bottom_Wall_Front_LIDAR_v1_1_body000.stl'
    triangles = vertices(path).reshape(-1, 3, 3) * .001 @ CAD_ROTATION.T
    segments = []
    for tri in triangles:
        points = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            low, high = tri[a, 2], tri[b, 2]
            if (low-z)*(high-z) < 0:
                fraction = (z-low)/(high-low)
                points.append(tri[a, :2]+fraction*(tri[b, :2]-tri[a, :2]))
        if len(points) == 2:
            segments.append(points)
    return np.asarray(segments)


def blocked_beams(origin, segments, angles=ANGLES):
    start, end = segments[:, 0, :], segments[:, 1, :]
    edges = end-start
    offset = start-origin
    cross = lambda a, b: a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]
    result = []
    for angle in angles:
        direction = np.array([np.cos(angle), np.sin(angle)])
        denominator = cross(direction, edges)
        valid = np.abs(denominator) > 1e-9
        distance = np.divide(cross(offset, edges), denominator,
                             out=np.full(len(edges), np.inf), where=valid)
        fraction = np.divide(cross(offset, direction), denominator,
                             out=np.full(len(edges), np.inf), where=valid)
        result.append(bool(np.any((distance > .001) & (fraction >= 0) & (fraction <= 1))))
    return np.asarray(result)


def clear_aperture_degrees(origin, segments):
    angles = np.deg2rad(np.linspace(-105, 105, 21001))
    blocked = blocked_beams(origin, segments, angles)
    center = len(angles)//2
    left = center
    right = center
    while left > 0 and not blocked[left-1]:
        left -= 1
    while right+1 < len(angles) and not blocked[right+1]:
        right += 1
    return float(np.rad2deg(angles[right]-angles[left]))


def test_optical_origin_is_inside_slit_at_deepest_clear_front_aperture():
    local = np.asarray(CONFIG['position_body_m'])
    origin = (CAD_ROTATION @ local)[:2]
    segments = panel_cross_section(local[2])
    path = SOURCE.parent / 'meshes/0029_Bottom_Wall_Front_LIDAR_v1_1_body000.stl'
    outer_front = float(np.max((vertices(path)*.001 @ CAD_ROTATION.T)[:, 0]))
    assert .025 < outer_front-origin[0] < .029
    assert not blocked_beams(origin, segments).any()
    assert abs(clear_aperture_degrees(origin, segments)-180) <= .03
    assert clear_aperture_degrees(origin-np.array([.001, 0]), segments) < 179.5
    assert clear_aperture_degrees(origin+np.array([.001, 0]), segments) > 180.5
