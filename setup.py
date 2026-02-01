from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="local-cp",
    version="0.1.0",
    author="Yifan Yu, Cheuk Hin Ho, Yangshuai Wang",
    author_email="",
    description="Conformal Prediction for PyTorch regression models with CUDA/MPS/CPU support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RoyYu0509/local_cp",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
        ],
        "examples": [
            "matplotlib>=3.5.0",
            "pandas>=1.4.0",
            "tqdm>=4.64.0",
        ],
    },
    include_package_data=True,
    keywords="conformal-prediction uncertainty-quantification pytorch machine-learning deep-learning",
)
