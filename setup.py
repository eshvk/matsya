from distutils.core import setup

setup(
        name='matsya',
        version='0.1.0',
        author='Eshwaran',
        author_email='eshhwaran@utexas.edu',
        packages=['kmeans', 'cocluster'],
        url='http://pypi.python.org/pypi/matsya/',
        license='LICENSE.txt',
        description='Text mining routines.',
        long_description=open('README.md').read(),
)
