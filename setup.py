from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements/base.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="protein-foundation-benchmark",
    version="0.2.0",
    author="RafiOISS",
    author_email="",
    description="Benchmark framework for comparing pretrained protein foundation models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RafiOISS/protein-foundation-benchmark",
    project_urls={
        "Bug Tracker": "https://github.com/RafiOISS/protein-foundation-benchmark/issues",
        "Documentation": "https://github.com/RafiOISS/protein-foundation-benchmark/docs",
        "Source Code": "https://github.com/RafiOISS/protein-foundation-benchmark",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.12",
    install_requires=requirements,
    extras_require={
        "tensorflow": ["tensorflow>=2.13.0"],
        "proteinbert": ["proteinbert>=1.0.0"],
        "onnx": ["onnx>=1.14.0", "onnxruntime>=1.15.0"],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.5.0",
            "pre-commit>=3.5.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
        ],
        "all": [
            "tensorflow>=2.13.0",
            "proteinbert>=1.0.0",
            "onnx>=1.14.0",
            "onnxruntime>=1.15.0",
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.5.0",
            "pre-commit>=3.5.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "protein-benchmark=src.framework.benchmark:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)