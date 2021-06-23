from setuptools import setup, find_packages

setup(
    name="fame",
    version="0.0.10-beta",
    description="FAME: A Framework for Large-scale Topic Modeling of Text Corpora with Neural Networks",
    url="https://github.com/shayanfazeli/fame",
    author="Shayan Fazeli",
    author_email="shayan.fazeli@gmail.com",
    license="Apache",
    classifiers=[
          'Intended Audience :: Science/Research',
          #'Development Status :: 1 - Beta',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Artificial Intelligence'
      ],
    keywords="machine learning,coronavirus,deep learning,inference",
    packages=find_packages(),
    python_requires='>3.6.0',
    scripts=[],
    install_requires=[],
    zip_safe=False
)
