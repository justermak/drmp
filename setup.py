from setuptools import setup, find_packages
from codecs import open
from os import path
import re


# Read version without importing the package
def get_version():
    version_file = path.join(path.dirname(__file__), "mpd", "__init__.py")
    with open(version_file, "r", encoding="utf-8") as f:
        content = f.read()
        version_match = re.search(
            r"^__version__\s*=\s*['\"]([^'\"]*)['\"]", content, re.M
        )
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")


ext_modules = []

here = path.abspath(path.dirname(__file__))
requires_list = []
with open(path.join(here, "requirements.txt"), encoding="utf-8") as f:
    for line in f:
        requires_list.append(str(line))


setup(
    name="mpd",
    version=get_version(),
    description="Motion Planning Diffusion",
    author="Joao Carvalho",
    author_email="joao@robots-learning.de",
    packages=find_packages(where=""),
    install_requires=requires_list,
)
