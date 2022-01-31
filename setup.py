import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mzprojection",
    version="0.0.2",
    install_requires=[
        "numpy",
        "scipy",
        "time",
    ],
    author="Shinya Maeyama",
    author_email="smaeyama@p.phys.nagoya-u.ac.jp",
    description="mzprojection package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/smaeyama/mzprojection",
    project_urls={
       "Bug Tracker": "https://github.com/smaeyama/mzprojection/issues",
    },
    classifiers=[
       "Programming Language :: Python :: 3",
       "License :: OSI Approved :: MIT License",
       "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
