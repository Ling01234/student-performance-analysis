from setuptools import find_packages, setup
from typing import List


def get_requirements(filepath: str) -> List[str]:
    """
    Returns the list of requirements

    Args:
        filepath (str): filepath of requirements

    Returns:
        List[str]: required packages
    """
    with open(filepath, 'r') as file:
        requirements = file.readlines()
        requirements = [req.replace('\n', '') for req in requirements]

        if '-e .' in requirements:
            requirements.remove('-e .')

    return requirements


setup(
    name='ml project',
    version='0.0.1',
    author='Ling',
    author_email='lzhang133@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
