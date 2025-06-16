from setuptools import setup, find_packages

setup(
    name="talia-ai-daemon",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "grpcio>=1.60.0",
        "grpcio-tools>=1.60.0",
        "transformers>=4.37.2",
        "torch>=2.2.0",
        "python-dotenv>=1.0.0",
        "urllib3==1.26.20",
    ],
    python_requires=">=3.8",
    author="Talia Project",
    author_email="your.email@example.com",
    description="AI daemon for text classification and summarization",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Talia-Project/ai-daemon",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 