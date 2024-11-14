from setuptools import setup, find_packages

setup(
    name='xzxTool',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "transformers",
        "torch",
        "numpy",
        "pandas",
    ],
    entry_points={
        'console_scripts': [
        ],
    },
    # 其他元数据，例如作者、描述等
    author='Zhixin Xie',
    description='For self use',
)