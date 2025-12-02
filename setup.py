import re
from codecs import open
from os import path

from setuptools import find_packages, setup


def get_version():
    version_file = path.join(path.dirname(__file__), "drmp", "__init__.py")
    with open(version_file, "r", encoding="utf-8") as f:
        content = f.read()
        version_match = re.search(
            r"^__version__\s*=\s*['\"]([^'\"]*)['\"]", content, re.M
        )
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")


ext_modules = []

requires_list = []
with open(path.join(path.dirname(__file__), "requirements.txt"), encoding="utf-8") as f:
    requires_list = list(map(str, f.readlines()))


setup(
    name="drmp",
    version=get_version(),
    description="drmp",
    packages=find_packages(where=""),
    install_requires=requires_list,
)
