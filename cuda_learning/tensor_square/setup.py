from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name = 'tensor_square',
    ext_modules = [
        CUDAExtension(
            name = 'tensor_square',
            sources = ['tensor_square.cpp', 'tensor_square_kernel.cu'],
        )
    ],
    cmdclass = {
        'build_ext': BuildExtension
    }
)