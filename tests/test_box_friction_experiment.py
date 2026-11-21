import copy
import json

import numpy as np
import pytest

from examples.box_friction_experiment import scaled_scene, outcome_comparison
from examples.box_random_dataset import DEFAULT_PLAN, cells, random_candidate
from examples.box_pickup import scene_model


def base_scene():
    plan = json.loads(DEFAULT_PLAN.read_text())
    return random_candidate(plan, cells(plan)[0], 'train', 0, 123)


def test_ridged_pad_experiment_only_changes_grip_friction_and_provenance():
    original = base_scene()
    saved = copy.deepcopy(original)
    changed = scaled_scene(original, 2.)
    assert original == saved
    assert changed['grasp']['sliding_friction'] == 2*original['grasp']['sliding_friction']
    assert .7 <= changed['grasp']['sliding_friction'] <= .8
    assert changed['table'] == original['table']
    assert changed['box_pose'] == original['box_pose']
    assert changed['seed'] == original['seed'] and changed['split'] == original['split']
    restored = copy.deepcopy(changed)
    restored['grasp']['sliding_friction'] = original['grasp']['sliding_friction']
    restored['domain_randomization']['physics_note'] = original['domain_randomization']['physics_note']
    del restored['contact_experiment']
    assert restored == original
    for invalid in (0, 1, -1, 5, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            scaled_scene(original, invalid)


def test_compiled_contact_friction_changes_without_changing_table_or_grip_force():
    config = base_scene()
    changed = scaled_scene(config, 2.)
    before, _ = scene_model(config, display_only=True)
    after, _ = scene_model(changed, display_only=True)
    index = after.geom('pickup_box_geom').id
    assert after.geom_friction[index, 0] == pytest.approx(2*before.geom_friction[index, 0])
    assert after.geom_priority[index] == 1
    np.testing.assert_array_equal(before.pair_friction, after.pair_friction)
    pair = after.pair('box_table_contact').id
    assert after.pair_friction[pair, 0] == config['table']['sliding_friction']
    for name in ('body_mass', 'body_inertia', 'geom_size', 'actuator_forcerange', 'actuator_ctrlrange',
                 'actuator_gainprm', 'actuator_biasprm'):
        np.testing.assert_array_equal(getattr(before, name), getattr(after, name))
    after.geom_friction[index, 0] = before.geom_friction[index, 0]
    np.testing.assert_array_equal(before.geom_friction, after.geom_friction)


def test_paired_comparison_counts_improvements_and_regressions_separately():
    before = [dict(episode_id=str(i), passed=passed, failed_checks=[] if passed else ['slipped'])
              for i, passed in enumerate((True, True, False, False))]
    after = [dict(episode_id=str(i), passed=passed, failed_checks=[] if passed else ['alignment'])
             for i, passed in enumerate((True, False, True, False))]
    result = outcome_comparison(before, after)
    assert result['summary'] == dict(attempts=4, baseline_successful=2, successful=2,
        net_success_change=0, improved=1, regressed=1, still_successful=1, still_failed=1)
    assert result['baseline_failed_checks'] == dict(slipped=2)
    assert result['failed_checks'] == dict(alignment=2)
    with pytest.raises(ValueError, match='same unique'):
        outcome_comparison(before, after[:-1])
