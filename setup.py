import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

with open("requirements.txt", "r") as fh:
    requirements = fh.read()

setup(
    name='gym_aigar',
    version = "0.0.2",
    author = "Anton Wiehe",
    author_email = "antonwiehe@gmail.com",
    description = ("Gym envs that replicate the aigar.io game to a certain extent."),
    license = "MIT",
    keywords = "reinforcement learning, environments, gym",
    url = "https://github.com/NotNANtoN/gym_aigar",
    packages = ['gym_aigar'],
    install_requires = requirements,
    long_description = read('README.md'),
    classifiers = [
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)

