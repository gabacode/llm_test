from setuptools import setup, find_packages

setup(
    name='llm_test',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'pydantic>=1.8',
    ],
    include_package_data=True,
)
