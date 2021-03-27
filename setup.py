import os
import setuptools
import glob

__version__ = '0.0.1dev2'

def _parse_requirements(path):
    with open(path) as f:
        return [
            line.rstrip()
            for line in f
            if not (line.isspace() or line.startswith('#'))
        ]

requirements = _parse_requirements('requirements.txt')
print(requirements)

setuptools.setup(
    name='onapy',
    version=__version__,
    url='https://github.com/xiong-jie-y/onapy',
    description='Onapy is the library to for next generation of masturbation.',
    author='xiong jie',
    author_email='fwvillage@gmail.com',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    install_requires=requirements,
    license='MIT License',
    scripts=['scripts/recognize_onaho_motion', 'scripts/recognize_waist_motion'],
    keywords='perception,masturbation',
    package_data={
        "": ["configs/*.*"]
    }
)
