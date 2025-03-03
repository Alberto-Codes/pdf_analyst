from setuptools import setup, find_packages

setup(
    name="pdf_analyst",
    version="0.1.0",
    description="A tool for extracting structured information from PDF documents using OCR and LLM",
    author="pdf_analyst Team",
    author_email="you@example.com",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "google-genai",
        "google-api-core",
        "google-cloud-core",
        "pydantic-graph",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.13",
    ],
) 