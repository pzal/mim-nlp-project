from setuptools import setup, find_packages

setup(
    name='matryoshka_experiment',  # Replace with your package name
    version='0.1.0',  # Initial version, update as needed
    description='A brief description of your package',  # Replace with a short description
    packages=find_packages(where='src'),  # Automatically find packages in the directory
    package_dir={'': 'src'},  # Specify the root package directory
    python_requires='>=3.10',  # Specify the minimum Python version required
    install_requires=[
        "accelerate",
        "torch",
        "datasets",
        "transformers",
        "sentence_transformers==3.0.1",
        "tqdm",
        "pandas",
        "scikit-learn",
        "neptune",
        "mteb"
    ]

)
