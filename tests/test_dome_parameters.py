"""The fixed dome parameters preserve the reviewed fit and source provenance."""
import json

import numpy as np

from sourccey.build import DOME_COLLISION
from tools.fit_dome import fit_dome


def test_offline_fit_reproduces_fixed_parameters():
    stored = json.loads(DOME_COLLISION.read_text())
    fitted = fit_dome()
    for key in ("center_base_m", "radius_m", "fit_rms_m"):
        np.testing.assert_allclose(fitted[key], stored[key], rtol=0, atol=1e-12)
    assert fitted["source_urdf_sha256_lf"] == stored["source_urdf_sha256_lf"]
    assert fitted["source_mesh_sha256"] == stored["source_mesh_sha256"]
