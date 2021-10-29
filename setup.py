"""Deli is a one-stop shop for all your sandwich (variance) needs. Specifically, `deli` implements a general API
for M-estimation. Estimating equations are both pre-built and can be custom-made. For help on creating custom
estimating equations, be sure to check out the ReadTheDocs documentation.

To install the deli library, use the following command

$ python -m pip install deli

"""

from setuptools import setup

exec(compile(open('deli/version.py').read(),
             'deli/version.py', 'exec'))


setup(name='deli',
      version=__version__,
      description='Generalized M-Estimation',
      keywords='m-estimation sandwich-variance estimating-equations',
      packages=['deli',
                ],
      include_package_data=True,
      license='MIT',
      author='Paul Zivich',
      author_email='zivich.5@gmail.com',
      url='https://github.com/pzivich/Deli',
      classifiers=['Programming Language :: Python :: 3.6',
                   'Programming Language :: Python :: 3.7',
                   'Programming Language :: Python :: 3.8',
                   'Programming Language :: Python :: 3.9',
                   'Programming Language :: Python :: 3.10',
                   ],
      install_requires=['numpy',
                        'scipy',
                        ],
      )
