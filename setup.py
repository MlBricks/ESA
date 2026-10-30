"""Optional native build for MLBricks ESA.

ESA_SKIP_NATIVE_BUILD=1 builds a Python-only wheel.
"""
import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME

os.environ.setdefault("MAX_JOBS", "1")
kwargs = dict(
    name="esa._C",
    sources=["esa/csrc_core/ops.cpp", "esa/csrc_core/ops_cpu.cpp"],
    extra_compile_args={"cxx": ["-O3", "-std=c++17"]},
)
if os.environ.get("ESA_SKIP_NATIVE_BUILD", "0").lower() in ("1", "true", "yes"):
    setup(ext_modules=[], include_package_data=True, zip_safe=False)
else:
    if CUDA_HOME is not None and torch.version.cuda is not None:
        kwargs["sources"].append("esa/csrc_core/ops_cuda.cu")
        kwargs["define_macros"] = [("WITH_CUDA", None)]
        kwargs["extra_compile_args"]["nvcc"] = ["-O3", "--use_fast_math", "-lineinfo"]
        kwargs["libraries"] = ["cublas"]
        ext = CUDAExtension(**kwargs)
    else:
        ext = CppExtension(**kwargs)
    setup(ext_modules=[ext], cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
          include_package_data=True, zip_safe=False)
