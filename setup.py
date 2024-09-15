from setuptools import setup, find_packages

setup(
	name='torchfit',
	version='0.1.0',
	description='A wrapper for using PyTorch models with Scikit-learn interface',
	packages=find_packages(),
	install_requires=[
		'numpy==1.26.4',
		'scikit-learn==1.5.2',
		'torch==2.2.2',
	],
	python_requires='>=3.7',
)
