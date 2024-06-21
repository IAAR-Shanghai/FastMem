from setuptools import setup, find_packages

setup(
    name='FastMem',
    version='1.0',
    packages=find_packages(),
    install_requires=[
      'regex',
      'tqdm',
      'torch==2.2.2',
      'torchaudio==2.2.2',
      'torchvision==0.17.2',
      'memory_profiler',
      'transformers==4.40',
      'termcolor',
      'accelerate',
    ],
)

