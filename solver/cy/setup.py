import os, glob
import numpy as np
from setuptools import setup
from Cython.Build import cythonize
HERE = os.path.dirname(os.path.abspath(__file__))
mods = sorted(glob.glob(os.path.join(HERE, "*.pyx")))
mods = [m for m in mods if not os.path.basename(m).startswith("_")]
setup(name="atsp_cy",
      ext_modules=cythonize(mods, language_level="3"),
      include_dirs=[np.get_include()])
