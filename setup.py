from setuptools import setup, find_packages

setup(
    name="synthhack-fraud-detection",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "joblib>=1.0.0",
        "xgboost>=1.4.0",
        "faker>=8.0.0",
        "imbalanced-learn>=0.8.0",
    ],
    python_requires=">=3.8",
    author="SynthHack Team",
    description="Real-time fraud detection system for UPI and card payments",
) 