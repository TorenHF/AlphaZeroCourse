from distutils import setup, find_packages

setup(
    name='your_package_name',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch'
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='A package that installs numpy and torch with Python standard libraries math and random.',
    url='https://your-package-url.example.com',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
