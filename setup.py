from glob import glob
import os
import sys
import setuptools

if sys.version_info[:2] < (3, 8):
    error = (
        "MFNetsSurrogates requires Python 3.8 or later (%d.%d detected). \n"
    )
    sys.stderr.write(error + "\n")
    sys.exit(1)

platforms = ["Linux", "Mac OSX"]
keywords = [
    "Networks",
    "Graph Theory",
    "Mathematics",
    "Uncertainty Quantification",
    "Multifidelity Modeling",
]

url = "https://www.alexgorodetsky.com/mfnets_surrogate_code/net.html"
project_urls = {
    # "Bug Tracker": "https://github.com/networkx/networkx/issues",    
    "Documentation": "https://www.alexgorodetsky.com/mfnets_surrogate_code/net.html",
    "Source Code": "https://github.com/goroda/MFNetsSurrogates",
}

with open("mfnets_surrogates/__init__.py") as fid:
    for line in fid:
        if line.startswith("__version__"):
            version = line.strip().split()[-1][1:-1]
            break

packages = [
    "mfnets_surrogates",
]

# add the tests subpackage(s)
package_data = {
    "mfnets_surrogates": ["tests/*.py"],
}

with open("README.org") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mfnets_surrogates",
    version=version,
    author="Alex A. Gorodetsky",
    author_email="goroda@umich.edu",
    description="A set of routines to enable construction of completely unstructured multifidelity surrogate models for fusing multiple information sources",
    long_description=long_description,
    url="https://github.com/goroda/MFNetsSurrogates",
    project_urls=project_urls,
    keywords=keywords,
    platforms=platforms,
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=packages,
    package_data=package_data,
    setup_requires=[],
    install_requires=['numpy >= 1.14','networkx'], # 'gslearn', 'pyro-ppl', 'torch', scipy, pandas],
    license='MIT',
)
