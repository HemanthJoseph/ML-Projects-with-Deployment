from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .' # we use this in requirements.txt so that it automatically calls setup.py,
# but we dont need it to be installed as well


#function takes file path and returns a list
#This list contains all pkgs to be installed
def get_requirements(file_path:str) -> List[str]:
    """
    This function will return the list of requirements
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines() #read all lines, it will read the new line character \n as well
        requirements = [req.replace("\n", "") for req in requirements] #removes that \n character

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    
    return requirements
        

setup(
name="mlproject",
version='0.0.1',
author="Hemanth Joseph Raj",
author_email="hemanth1@umd.edu",
packages=find_packages(),
install_requires=get_requirements('requirements.txt')
)