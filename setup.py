from skbuild import setup
import re


def get_project_property(prop):
    '''Helper function to read property from __init__.py'''
    with open('fmf2/__init__.py', mode='r') as fil:
        result = re.search(r'{!s}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop), fil.read())
        return result.group(1)


setup(
    name="fmf2",
    version=get_project_property('__version__'),
    description="Fast Matched Filter implementation with SYCL backend",
    author='jorgen@nordmoen.net',
    license="MIT",
    packages=['fmf2'],
    python_requires=">=3.7",
    install_requires=['numpy'],
    extras_require={
        'tests': ['pytest', 'pytest-benchmark'],
    },

)
