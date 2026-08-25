from pathlib import Path
import os, torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME
ROOT=Path(__file__).parent; os.environ.setdefault("MAX_JOBS","1")
c=ROOT/"esa"/"csrc_core"; sources=[str(c/"ops.cpp"),str(c/"ops_cpu.cpp")]; kwargs=dict(name="esa._C",sources=sources,extra_compile_args={"cxx":["-O3","-std=c++17"]})
if CUDA_HOME is not None and torch.version.cuda is not None:
 kwargs["sources"].append(str(c/"ops_cuda.cu")); kwargs["define_macros"]=[("WITH_CUDA",None)]; kwargs["extra_compile_args"]["nvcc"]=["-O3","--use_fast_math","-lineinfo"]; kwargs["libraries"]=["cublas"]; ext=CUDAExtension(**kwargs)
else: ext=CppExtension(**kwargs)
setup(ext_modules=[ext],cmdclass={"build_ext":BuildExtension.with_options(use_ninja=True)},include_package_data=True,zip_safe=False)
