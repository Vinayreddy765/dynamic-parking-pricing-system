from setuptools import setup, find_packages

setup(
    name="dynamic-parking-pricing",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "bokeh>=2.4.0",
        "pathway-python>=0.7.0",
        "plotly>=5.0.0"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Dynamic Pricing System for Urban Parking Lots",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/[your-username]/dynamic-parking-pricing-system",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)