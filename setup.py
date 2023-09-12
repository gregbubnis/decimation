from setuptools import setup
from setuptools import find_packages

setuptools.setup(
    name="decimate",
    version="0.1",
    author="Greg Bubnis",
    description="blocky terrain visualization",
    long_description_content_type=open('README.md').read(),
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'scipy',
        'pandas',
        'open3d==0.16',
        'numpy',
        'jupyter',
        'numpy-stl'
      ], 
    dependency_links=[],
    python_requires='>=3.6',
)