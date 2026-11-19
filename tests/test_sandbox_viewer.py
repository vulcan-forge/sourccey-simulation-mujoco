from types import SimpleNamespace

from mujoco.studio import messages
from mujoco.studio.dear_imgui import dear_imgui as imgui

from sourccey.sandbox_viewer import ManualInput, WorkshopAction, WorkshopPanel


def panel_with_keys(monkeypatch, *, down=(), pressed=(), editing=False):
    panel = WorkshopPanel()
    outgoing = []
    panel.viewer = SimpleNamespace(send_to_sim=outgoing.append)
    monkeypatch.setattr(imgui, "GetIO", lambda: SimpleNamespace(
        WantCaptureKeyboard=editing, KeyCtrl=False, KeyAlt=False, KeySuper=False, KeyShift=False))
    monkeypatch.setattr(imgui, "IsKeyDown", lambda key: key in down)
    monkeypatch.setattr(imgui, "IsKeyPressed", lambda key, repeat: key in pressed)
    return panel, outgoing


def test_held_keys_generate_combined_commands_and_release_stops(monkeypatch):
    panel, sent = panel_with_keys(monkeypatch, down=(imgui.Key.W, imgui.Key.R, imgui.Key.X))
    panel.update(messages.UpdateEvent())
    controls = sent[-1].controls
    assert controls.drive == (1, 0, 0)
    assert controls.hand == (0, 0, 1)
    assert controls.grip == 1
    monkeypatch.setattr(imgui, "IsKeyDown", lambda key: False)
    panel.update(messages.UpdateEvent())
    assert sent[-1].controls.drive == (0, 0, 0)
    assert sent[-1].controls.hand == (0, 0, 0)


def test_ui_editing_suppresses_robot_input(monkeypatch):
    panel, sent = panel_with_keys(monkeypatch, down=(imgui.Key.W,), editing=True)
    panel.update(messages.UpdateEvent())
    assert sent[-1].controls.drive == (0, 0, 0)


def test_tab_changes_arm_and_escape_requests_clean_shutdown(monkeypatch):
    panel, sent = panel_with_keys(monkeypatch, pressed=(imgui.Key.Tab, imgui.Key.Escape))
    panel.update(messages.UpdateEvent())
    assert panel.side == "right"
    assert any(isinstance(event, WorkshopAction) and event.action == "quit" for event in sent)
    assert isinstance(sent[-1], ManualInput)
    assert sent[-1].controls.side == "right"
