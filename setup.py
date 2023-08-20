from setuptools import setup

setup(
    name='news-summarization',
    version='0.1.0',
    py_modules=['news-summarization'],
    install_requires=[
        'Click',
    ],
    entry_points={
        'console_scripts': [
            'news-summarization = news-summarization:cli',
        ],
    },

)