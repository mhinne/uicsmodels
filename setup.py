from setuptools import setup, find_packages

setup(
    name='Prob-models',
    version='0.0.1',
    url='https://github.com/mhinne/prob-models',
    author='Max Hinne',
    author_email='max.hinne@donders.ru.nl',
    packages=find_packages(),
    install-requires=[
        'jax',
        'jaxtyping',
        'distrax @ git+https://github.com/deepmind/distrax.git',
        'jaxkern @ git+https://github.com/JaxGaussianProcesses/JaxKern.git',
        'blackjax @ git+https://github.com/Hesterhuijsdens/blackjax.git'
    ]

)
