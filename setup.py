from setuptools import setup
import emgineer

DESCRIPTION = 'emgineer: emg processing tool'
NAME = 'emgineer'
AUTHOR = 'Naoki Hashimoto'
AUTHOR_EMAIL = 'naokih1998@yahoo.co.jp'
URL = 'https://github.com/mechAneko12/emgineer'
LICENSE = 'MIT LICENSE'
DOWNLOAD_URL = 'https://github.com/mechAneko12/emgineer'
VERSION = emgineer.__version__
PYTHON_REQUIRES = '>=3.6'

INSTALL_REQUIRES = [
    'numpy>=1.21.5',
    'pandas>=1.3.5',
    'scikit-learn>=1.0.2',
    'matplotlib>=3.5.1'
]

PACKAGES = [
    'emgineer'
]

with open('README.md', 'r') as fp:
    readme = fp.read()
long_description = readme

setup(name=NAME,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      maintainer=AUTHOR,
      maintainer_email=AUTHOR_EMAIL,
      description=DESCRIPTION,
      long_description=long_description,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      python_requires=PYTHON_REQUIRES,
      install_requires=INSTALL_REQUIRES,
      # extras_require=EXTRAS_REQUIRE,
      packages=PACKAGES,
      # classifiers=CLASSIFIERS
    )