from setuptools import setup, find_packages

setup(
    name='sdp-detection-pipeline',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'argparse',
        'numpy',
        'scipy',
        'matplotlib',
        'biopython',
        'scikit-learn',
        'pandas',
        'fastcluster',
        'prince',
        'logomaker',
        'wordcloud',
        'nltk',
    ],
    entry_points={
        'console_scripts': [
            'sdp-pipeline = pipeline:main',
        ]
    },
)
