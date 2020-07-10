from setuptools import setup, find_packages

setup(name='ai-ml-common',
      version='3.7',
      description='A Basic setup for a common library for AI and ML',
      author='Rohit Gopalan',
      author_email='rohitgopalan1990@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
        'numpy', 'pandas', 'scikit-learn', 'tensorflow', 'matplotlib'
      ],
      zip_safe=False)
