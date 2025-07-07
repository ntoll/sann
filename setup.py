from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")
setup(
    name="sann",
    version="0.0.1",
    description="A Simple Artificial Neural Network (SANN) module, for educational use.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ntoll/sann",
    author="Nicholas H.Tollervey",
    author_email="ntoll@ntoll.org",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: Implementation :: MicroPython",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="education, artificial intelligence, neural networks",
    py_modules=["sann"],
    python_requires=">=3.8, <4",
    project_urls={
        "Bug Reports": "https://github.com/ntoll/sann/issues",
        "Source": "https://github.com/ntoll/sann/",
    },
)
