from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="llm-pruning-framework",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A unified framework for pruning Large Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/llm_prune",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "llm-prune=main:main",
        ],
    },
    keywords="llm, pruning, compression, neural networks, transformers",
    project_urls={
        "Bug Reports": "https://github.com/your-username/llm_prune/issues",
        "Source": "https://github.com/your-username/llm_prune",
        "Documentation": "https://github.com/your-username/llm_prune#readme",
    },
)