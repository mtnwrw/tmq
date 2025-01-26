# vim: set ts=4 sw=4 :
# _*_ coding: utf-8 _*_
# Copyright (c) 2025 TMQ Authors
# SPDX-License-Identifier: MPL-2.0

__author__ = 'Martin Wawro'

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

setup(
    name="tmq",
    package_dir={"" : "tmq"},
    ext_modules=[
        CUDAExtension("tmq_cuda", [
            "tmq/tmq_cuda.cu"
        ]),
        CppExtension("tmq_native", [
            "tmq/tmq_native.cpp"
        ]),
    ],
    cmdclass={ "build_ext" : BuildExtension})
