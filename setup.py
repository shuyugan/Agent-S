from setuptools import setup, find_packages

setup(
    name='agent_s',
    version='0.1.0',
    description='A library for creating general purpose GUI agents using multimodal LLMs.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Saaket Agashe',
    author_email='saagashe@ucsc.edu',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'pandas',
        'openai',
        'transformers',
        'fastapi',
        'openaci @ git+https://github.com/simular-ai/OpenACI.git@main#egg=openaci',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)