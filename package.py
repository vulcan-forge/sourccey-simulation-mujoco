"""Build the Windows launcher, then update the root copy when it is not running."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

root = Path(__file__).resolve().parent
base = Path(sys.base_prefix)
env = os.environ.copy()
env['TCL_LIBRARY'] = str(base / 'tcl' / 'tcl8.6')
env['TK_LIBRARY'] = str(base / 'tcl' / 'tk8.6')
subprocess.run(
    [sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean',
     '--distpath', str(root / 'dist'), '--workpath', str(root / 'build' / 'pyinstaller'),
     str(root / 'SourcceyMuJoCo.spec')],
    cwd=root, env=env, check=True,
)
fresh = root / 'dist' / 'SourcceyMuJoCo.exe'
try:
    shutil.copy2(fresh, root / fresh.name)
except PermissionError:
    print(f'Built {fresh}. Close the running SourcceyMuJoCo.exe, then rerun package.cmd '
          'to update the root copy.')
else:
    print(f'Updated {root / fresh.name}')
