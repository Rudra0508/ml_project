from setuptools import find_packages, setup 
from typing import List
def get_requirements(file_path)->List[str]:
    """This function will return the list of requirements"""
    requirements=[]
    Hyphen_E_Dot='-e .'
    with open(file_path)as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        if Hyphen_E_Dot in requirements:
            requirements.remove(Hyphen_E_Dot)

    return requirements
setup(
    name="ml_project",
    version='0.0.1',
    author ="Rudra",
    author_email="rudrapahua@gmail.com",
    packages=find_packages(),
    install_requries=get_requirements('requirements.txt')
)