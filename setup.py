from distutils.core import setup

setup(
        name='Matsya',
        version='0.1.0',
        author='Eshwaran',
        author_email='eshhwaran@utexas.edu',
        packages=['kmeans', 'cocluster'],
        url='http://pypi.python.org/pypi/Matsya/',
        license='LICENSE.txt',
        description='Machine learning for text.',
        long_description=open('README.md').read(),
)
