'''Build package.'''

import os
from setuptools import find_packages, setup

DESCRIPTION = 'Python library for converting PDF to Word document (docx).'
EXCLUDE_FROM_PACKAGES = ["build", "dist", "test"]


def get_version():
    '''Return version number.'''
    return '1.0.0'

def load_long_description(fname="README.md"):
    '''Load README.md for long description'''
    if os.path.exists(fname):
        with open(fname, "r", encoding="utf-8") as f:
            long_description = f.read()
    else:
        long_description = DESCRIPTION

    return long_description

def load_requirements(fname="requirements.txt"):
    '''Load requirements.'''
    ret = list()
    if os.path.exists(fname):
        with open(fname) as f:
            for line in f:
                ret.append(line.strip())
    return ret


setup(
    name="pdf2word",
    version=get_version(),
    keywords=["pdf-to-word", "pdf-to-docx"],
    description=DESCRIPTION,
    long_description=load_long_description(),
    long_description_content_type="text/markdown",
    license="GNU AFFERO GPL 3.0",
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/pdf2word',
    packages=find_packages(exclude=EXCLUDE_FROM_PACKAGES),
    include_package_data=True,
    zip_safe=False,
    install_requires=load_requirements(),
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            "pdf2word=pdf2word.main:main"
            ]},
) 