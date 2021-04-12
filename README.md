# sfTA
README for sfTA code developed for use with VASP CCSD calculations - Tina Mihm, April 8, 2021

Directory contains the following files:
 - Na32_Data.csv - Structure factor data for 100 MP2 calculations for a Na 32 electron system
 - TwistAngles.csv - twist angles used to run the 100 MP2 calcualtions with the accociated python indexing
 - sfTA.py - Python code set up to use the Na32 strucutre factor data to obtain a special twist angle that reproduces twist averaged energies. 

System requirements: 
------------------------

The sfTA code is a python-based code that requires the following libraries to run:
  - python 3.0 or above 
  - numpy
  - pandas  
  - matplotlib  

Code has been tested and run using Python 3.7.3

Installation guide/Demo:
-------------------------
Code can be downloaded from the github repository and run right away. (Note: download times will varry) 
The code is run using the following command: $python3 sfTA.py

The following outputs are produced: 
  - csv for various data sets printed as .csv
  - graphs of the various structure factors, saved as .png, including: 
      - 100 MP2 strcture factors
      - 100 MP2 structure factors + twist averaged structure factor
      - 100 MP2 structure factors + twist averaged structure factor + sfTA structure factor
  - prints to screen the python index for the special twist angle ["S_G special twist angle (min diff, sfTA twist angle python index)" ]  - can be used in conjunction with TwistAngles.csv to find the special twist angle for sfTA 

Code should take about 20-30 seconds to run depending on data set size.

Instructions for use
--------------------------
1. Change the line: "Data = pd.read_csv("Li-bcc/k222/BasisSetData/Nband16/Li-bcc-k222-nb16_Data.csv")" to have it read in the desired data set csv
2. Update all paths for graphs and csv to save then to desired folder (e.g search for "Li-bcc" to locate paths in current code version - April 12, 2021) 
3. Save and run using above python command in terminal
4. Code should run and produce graphs and csv before printing special twist angle number to screen (see above)  
