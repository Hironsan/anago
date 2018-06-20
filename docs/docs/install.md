# Install anaGo

anaGo is compatible with 64-bit CPython 3.4+ and runs on Unix/Linux, macOS/OS X and Windows. The latest anaGo releases are available over pip. Installation requires a working build environment.

```bash
pip install -U anago
```

<!--## Installation instructions-->

## pip

To install anaGo, simply use [pip](https://pypi.org/project/pip/):

```bash
pip install anago
```

When using pip it is generally recommended to install packages in a [virtual environment](https://virtualenv.pypa.io/en/stable/) to avoid modifying system state:

```bash
virtualenv venv --python=python3
source venv/bin/activate
pip install anago
```

## Upgrading anaGo

When updating to a newer version of anaGo, it's generally recommended to start with a clean virtual environment. If you're upgrading to a new major version, make sure you have the latest compatible models installed, and that there are no old shortcut links or incompatible model packages left over in your environment, as this can often lead to unexpected results and errors. If you've trained your own models, keep in mind that your train and runtime inputs must match. This means you'll have to retrain your models with the new version.

```bash
pip install -U anago
```

## Install from source

The other way to install anaGo is to clone its GitHub repository and build it from source. That is the common way if you want to make changes to the code base. You'll need to make sure that you have a development environment consisting of a Python distribution including header files, a compiler, pip, virtualenv and git installed. The compiler part is the trickiest. How to do that depends on your system. See notes on Ubuntu, OS X and Windows for details.

or install from the repository:

```bash
git clone https://github.com/Hironsan/anago.git  # clone anaGo
cd anago                                         # navigate into directory

virtualenv venv --python=python3                 # create environment in venv
source venv/bin/activate                         # activate virtual environment
export PYTHONPATH=`pwd`                          # set Python path to anaGo directory
pip install -r requirements.txt                  # install all requirements
python setup.py install
```
