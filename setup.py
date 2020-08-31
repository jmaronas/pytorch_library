import setuptools


setuptools.setup(
    name="Pytorch Library", # Replace with your own username
    version="1.5.0",
    author="Juan Maroñas and Sergio Álvarez",
    author_email="jmaronasm@gmail.com",
    description="Just a common interface for pytorch utils to be bug safe and share our coding",
    long_description_content_type="text/markdown",
    url="https://github.com/jmaronas/pytorch_library",
    long_description=long_description,
    packages=setuptools.find_packages(),
    classifiers=[
                "Programming Language :: Python :: 3",
                "License :: OSI Approved :: MIT License",
                "Operating System :: OS Independent",
                ],
    python_requires='>=3.7',
    )
