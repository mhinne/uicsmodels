from setuptools import setup, find_packages

setup(
    name='uicsmodels',
    version='0.0.1',
    url='https://github.com/mhinne/uicsmodels',
    author='Max Hinne',
    author_email='max.hinne@donders.ru.nl',
    packages=find_packages(),
    install_requires=[
        'jax',
        'jaxtyping',
        'distrax @ git+https://github.com/deepmind/distrax.git@f6e656c',
        'jaxkern @ git+https://github.com/JaxGaussianProcesses/JaxKern.git',
        'blackjax @ git+https://github.com/Hesterhuijsdens/blackjax.git'
    ]

)
