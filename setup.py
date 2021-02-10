from setuptools import setup

setup(name='lima-tfner',
      version='0.1',
      description='Tensorflow NN model for NER',
      url='http://',
      license='CEA',
      packages=['tfner'],
      include_package_data=True,
      zip_safe=False,
      install_requires=['tensorflow<=1.9.0'])
