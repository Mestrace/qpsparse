import sys
from setuptools import setup, find_packages
from os import path
from io import open


BASE_DIR = path.abspath(path.dirname(__file__))

INSTALL_REQUIRED = [
    'pandas>=0.25.0',
    'tatsu>4.1.0',
    'numpy',
    'scipy'
]

with open(path.join(BASE_DIR, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Only include pytest-runner in setup_requires if we're invoking tests
if {'pytest', 'test', 'ptr'}.intersection(sys.argv):
    setup_requires = ['pytest-runner']
else:
    setup_requires = []

setup(
    name='qpsparse',
    version='0.0.1',
    description='QPSParse read and parses linear optimization problems in '
                'MPS / QPS format.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Yida Liu',
    author_email='yida.liu@case.edu',
    maintainer='Yida Liu',
    maintainer_email='yida.liu@case.edu',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    packages=find_packages(exclude=['contrib', 'docs', 'tests', 'testdata']),
    python_requires='>=3.0, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, <4',
    install_requires=INSTALL_REQUIRED,
    tests_require=['pytest'],
    setup_requires=['pytest-runner'],
    include_package_data=True,
    zip_safe=False
)
