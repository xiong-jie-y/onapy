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

from setuptools.command.install import install
from subprocess import getoutput

class PostInstall(install):
    pkgs = ' git+https://github.com/xiong-jie-y/remimi.git'\
           ' git+https://github.com/xiong-jie-y/mmdetection.git'
    def run(self):
        install.run(self)
        print(getoutput('pip install'+self.pkgs))

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
    },
    dependency_links=[
        'https://github.com/xiong-jie-y/remimi.git',
        'https://github.com/xiong-jie-y/mmdetection.git'
    ],
    cmdclass={'install': PostInstall}
)
