"""detector package: the YOLO object detector node.

Importing this package (it happens before detector_node.py runs) makes the project's virtual environment visible
when the node was started by the SYSTEM Python, e.g. after a plain 'colcon build' instead of 'python -m colcon build'.
Without it that node cannot import ultralytics and dies. Nothing changes when:
  - the node already runs inside a virtual environment,
  - ultralytics is already importable (installed system-wide, as in a Docker image), or
  - there is no virtual environment for this Python version.
The venv is looked for in $SLAM_VENV, else $SLAM_WS/venv, else ~/slam_ws/venv.
"""
import importlib.util
import os
import site
import sys

def project_venv():
    """Folder of the project's virtual environment (it may not exist)."""
    venv = os.environ.get("SLAM_VENV") or os.path.join(os.environ.get("SLAM_WS", "~/slam_ws"), "venv")
    return os.path.expanduser(venv)

def project_venv_site_packages():
    """The venv's site-packages for THIS Python version, or None (a venv of another version cannot be used)."""
    path = os.path.join(project_venv(), "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
    return path if os.path.isdir(path) else None

def use_project_venv():
    """Put the venv's packages on sys.path, where a venv Python would have them: after the entries ROS added
    (PYTHONPATH) and before the system ones. Returns the folder it added, or None when nothing was needed."""
    if sys.prefix != sys.base_prefix:                         # already running inside a virtual environment
        return None
    if importlib.util.find_spec("ultralytics") is not None:   # available as it is
        return None
    site_dir = project_venv_site_packages()
    if site_dir is None or site_dir in sys.path:
        return None
    user_site = site.getusersitepackages()
    position = next((i for i, p in enumerate(sys.path) if "dist-packages" in p or p == user_site), len(sys.path))
    sys.path.insert(position, site_dir)
    print(f"detector: this node runs under the system Python; using the packages of {project_venv()}", file=sys.stderr)
    return site_dir

use_project_venv()