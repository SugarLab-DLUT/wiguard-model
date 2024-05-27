from setuptools import setup, find_packages

setup(
    name='wifall',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'csiread==1.4.0',
        'numpy==1.26.4',
        'python-socketio==5.11.2',
        'toml==0.10.2',
        'aiohttp==3.9.5'
    ],
    entry_points={
        'console_scripts': [
            'wifall=main:main',
        ],
    },
)
