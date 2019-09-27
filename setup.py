import setuptools

with open("README.md",'r') as fh:
    long_description=fh.read()

setuptools.setup(
    name='parareal_dedalus',
    version='0.0.1',
    author='Andrew Clarke',
    author_email='scatc@leeds.ac.uk',
    description='Implementation of parareal algorithm in dedalus',
    long_description=long_description,
    url='https://gitlab.com/scatc/parareal_dedalus',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.4",
)
