from setuptools import setup, find_packages

setup(
    name="travel-guide-chatbot",
    version="0.1.0",
    description="A travel guide chatbot that uses RAG and Claude to provide travel information",
    author="ogokmen",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.4.2",
        "faiss-cpu>=1.7.4",  # Use faiss-gpu if GPU support is needed
        "sentence-transformers>=2.2.2",
        "transformers>=4.35.0",
        "numpy>=1.24.0",
        "requests>=2.31.0",
        "boto3>=1.28.0",
        "botocore>=1.31.0",
        "python-dotenv>=1.0.0",  # Added for environment variable management
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    entry_points={
        "console_scripts": [
            "travel-chatbot=src.api:app",
        ],
    },
) 