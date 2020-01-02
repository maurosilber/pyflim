from setuptools import setup, find_packages

setup(
    name='pyflim',
    version='0.0.0',
    packages=find_packages(),
    url='https://github.com/maurosilber/pyflim',
    license='MIT',
    author='Mauro Silberberg',
    author_email='maurosilber@gmail.com',
    install_requires=['numpy', 'numba']
)
