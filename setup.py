from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#") and not line.startswith("git+")]

setup(
    name="smart-logistics-delivery-optimization",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Smart Logistics & Delivery Optimization using Multi-Modal Routing + RL",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/smart-logistics-delivery-optimization",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=requirements,
    extras_require={
        "dev": [
            "pre-commit>=3.5.0",
            "black>=23.9.1",
            "flake8>=6.1.0",
            "pytest>=7.4.3",
            "mypy>=1.6.1",
            "isort>=5.12.0",
        ],
    },
    dependency_links=[
        "git+https://github.com/supremeloki/lono_libs.git#egg=lono_libs",
    ],
    entry_points={
        "console_scripts": [
            "logistics-check=check_all:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    license="MIT",
    python_requires=">=3.9",
)