import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    requirements = [req for req in requirements if req.split('~=')[0] not in ['gensim','six','fastdtw','tensorflow-probability','tensorflow-addons']]

setuptools.setup(
    name="relegy",
    version="0.0.1",
    author="Łukasz Brzozowski, Kacper Siemaszko",
    author_email="author@example.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lukaszbrzozowski/ReLeGy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    requirements=requirements
)
