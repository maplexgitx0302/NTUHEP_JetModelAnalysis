from setuptools import setup, find_packages

setup(
    name='JetModelAnalysis',
    version='1.0.0',
    description='A package for jet model analysis',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Yi-An Chen',
    author_email='maplexworkx0302@gmail.com',
    url='https://github.com/maplexgitx0302/HEP_JetModelAnalysis',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        # Common packages,
        'ipynb-py-convert',
        'matplotlib',
        'numpy',
        'pandas',
        'scikit-learn',
        'seaborn',

        # HEP related
        'awkward',
        'h5py',
        'uproot',
        'tables',

        # Machine Learning
        'torch',
        'torch_geometric',

        # ML Tools
        'lightning',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Physics :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
)
