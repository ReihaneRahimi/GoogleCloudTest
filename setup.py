from setuptools import setup, find_packages

#Setup parameters for Google Cloud ML Engine
setup(name = 'trainer',
      version = '0.1',
      packages = find_packages(),
      description = 'Example to run keras on gcloud ml-engine',
      author = 'Reihane Rahimilarki',
      author_email = 'supernova7749@gmail.com',
      license = None,
      install_requires = [
          'keras',
          'h5py'
      ],
      zip_safe = False)

