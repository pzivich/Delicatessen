"""Delicatessen is a one-stop shop for all your sandwich (variance) needs. `delicatessen` implements a general API for
the generalized calculus of M-estimation. Both pre-built and custom, user-specified estimating equations are compatible
with the M-Estimator. For an introduction to M-Estimation, using `delicatessen`, and building your own estimating
equations, see the documentation at https://deli.readthedocs.io/en/latest/ .

To install the `delicatessen`, use the following command

$ python -m pip install delicatessen

"""

from setuptools import setup

exec(compile(open('delicatessen/version.py').read(),
             'delicatessen/version.py', 'exec'))

with open("README.md") as f:
    descript = f.read()


setup(name='delicatessen',
      version=__version__,
      description='Generalized M-Estimation',
      keywords='m-estimation sandwich-variance estimating-equations',
      packages=['delicatessen',
                'delicatessen.estimating_equations'
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
      long_description=descript,
      long_description_content_type="text/markdown",
      )
