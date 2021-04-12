# sfTA
README for sfTA code developed for use with VASP CCSD calculations - Tina Mihm, April 8, 2021

Directory contains the following files:
 - Na32_Data.csv - Structure factor data for 100 MP2 calculations for a Na 32 electron system
 - TwistAngles.csv - twist angles used to run the 100 MP2 calcualtions with the accociated python indexing
 - sfTA.py - Python code set up to use the Na32 strucutre factor data to obtain a special twist angle that reproduces twist averaged energies. 

The sfTA code is a python-based code that requires the following libraries to run:
  - python 3.0 or above 
  - numpy
  - pandas  
  - matplotlib  

The code is run using the following command: $python3 sfTA.py

The following outputs are produced: 
  - csv for various data sets printed as .csv
  - graphs of the various structure factors including: 
      - 100 MP2 strcture factors
      - 100 MP2 structure factors + twist averaged structure factor
      - 100 MP2 structure factors + twist averaged structure factor + sfTA structure factor
  - prints to screen the python index for the special twist angle ["S_G special twist angle (min diff, sfTA twist angle python index)" ]  - can be used in conjunction with TwistAngles.csv to find the special twist angle for sfTA 
