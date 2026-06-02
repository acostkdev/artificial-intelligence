import ctypes
import os

_RTLD_GLOBAL = 0x00100

NV_LIBS = [
    "libcuda.so.1",
    "libcudart.so.12",
    "libcublas.so.12",
    "libcublasLt.so.12",
    "libcudnn.so.9",
    "libcufft.so.11",
    "libcurand.so.10",
    "libcusolver.so.11",
    "libcusparse.so.12",
    "libnccl.so.2",
    "libnvJitLink.so.12",
]

def enable():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(script_dir))
    venv = os.path.join(repo_root, ".venv")
    nvidia_dir = os.path.join(venv, "lib", "python3.13", "site-packages", "nvidia")
    sys_lib = "/lib/x86_64-linux-gnu"

    loaded = False
    for lib in NV_LIBS:
        paths = []
        if os.path.exists(os.path.join(sys_lib, lib)):
            paths.append(os.path.join(sys_lib, lib))
        if os.path.isdir(nvidia_dir):
            for pkg in os.listdir(nvidia_dir):
                p = os.path.join(nvidia_dir, pkg, "lib", lib)
                if os.path.exists(p):
                    paths.append(p)
        for p in paths:
            try:
                ctypes.CDLL(p, _RTLD_GLOBAL)
                loaded = True
                break
            except OSError:
                continue

    if loaded:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    return loaded
