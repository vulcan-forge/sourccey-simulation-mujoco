# PyInstaller one-file Windows app. Robot MJCF and source STL meshes live in
# the executable's private extraction directory and require no adjacent files.
from pathlib import Path
from PyInstaller.utils.hooks import collect_dynamic_libs

ROOT = Path(SPECPATH)
datas = [(str(ROOT / "models"), "models")]
binaries = []
hiddenimports = ["tkinter", "tkinter.ttk", "mujoco.viewer", "glfw"]
for package in ("mujoco", "glfw"):
    binaries += collect_dynamic_libs(package)

a = Analysis(
    [str(ROOT / "sourccey" / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "IPython", "matplotlib"],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="SourcceyMuJoCo",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
