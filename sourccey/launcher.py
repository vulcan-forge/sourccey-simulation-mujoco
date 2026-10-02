"""Double-click launcher for the frozen desktop application."""
import os
from pathlib import Path
import sys
import traceback


def main():
    try:
        # PyInstaller executes this file as the top-level script, outside its
        # package context, so use the package's absolute import path.
        from sourccey.app import main as run
        run()
    except Exception:
        details = traceback.format_exc()
        log_dir = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "SourcceyMuJoCo"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "error.log"
        log_path.write_text(details, encoding="utf-8")
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Sourccey MuJoCo", f"The simulator could not start. Details were saved to:\n{log_path}")
            root.destroy()
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
