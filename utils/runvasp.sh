#! /usr/bin/sh

# This is utility script designed to run VASP with CCSD.F integrated.
# This will be incorporated into public release and this file will be kept up to date 
# to work with the current version of VASP.
# Direct any questions to james-shepherd@uiowa.edu

####################
# USER DEFINED OPTIONS
####################


# The lines below need to be edited to provide paths to certain required files. 

# - An HPC environment ready to run VASP. The binary file is in the location below. 
VASP="/path_to_VASP/bin/vasp_std"

# - A POTCAR file with the potentials required for VASP.
POTCAR="/path_to_POTCAR/POTCAR"

# - A POSCAR file with the location of the atoms. 
POSCAR="/path_to_POSCAR/POSCAR"

# - A IBZKPT.4.J.1 file with the appropriate twist angles generated externally. 
# Two examples are provided:
# IBZKPT.4.J.1_MP2 for the 100 random twist angles
# IBZKPT.4.J.1_CCSD for the special twist angle
shiftfile="IBZKPT.4.J.1"

# - These cutoffs are recommended in the VASP POTCAR files and are parameters that ultimately 
# need converging. 
enc=80
egw=60

# Label for the system

name="Na"

# There are two options for how this script runs. Either this is meant to run 100 twist 
# angles or 1 CCSD twist angle. 

# - These two lines need to be uncommented to run MP2:
ccmaxit=1
mp2flag="CCALGO=MP2"
ccfilehandle="MP2"

# - These two lines need to be uncommented to run CCSD once the special twist angle is found:
#ccmaxit=100
#mp2flag=""
#ccfilehandle="CCSD"

####################
# Script begins here
####################

# Calculate the number of shifts
nshift=`wc -l $shiftfile | awk '{print $1;}'`
nshift=`python -c "e=($nshift); print e" `
echo "performing calculations for " $nshift " shifts."

# Counts the number of twist angles for file labels
x=1

# The line below loops over the lines in IBZKPT.4.J.1
for ishift in `seq 4 1 $nshift`
do

mkdir shift.$ishift
cd shift.$ishift
cp $POTCAR .
cp $POSCAR .
cp ../$shiftfile .

# The line below loops over different k-point grids. Here, we are only interested in a single 2x2x2 grid. 
# 
for k in 2
do

# Sets up directories
# 
mkdir sweepTA.kp.$k
cd    sweepTA.kp.$k
cp $POTCAR .
cp $POSCAR .
cp ../$shiftfile .


shift=`cat $shiftfile | sed -n $ishift'p' | awk '{print $1 " " $2 " " $3;}'`

# Generates the file that is read in for VASP for the k-point mesh
# 
cat >KPOINTS <<!
Automatically generated mesh
       0
Gamma
 $k $k $k
 $shift
!

echo $k

# This is a label for a volume ID and is just set to 1 here
# 
for fac in "1" 
do

rm WAVECAR
rm CHG*

# DFT ground state calculation. Produces a WAVECAR containing orbital 
# coefficients used as better starting point in HF calculation
# 
cat >INCAR <<!
ENCUT=$enc
NBANDS=32
System = $name
EDIFF  = 0.1E-06
ISYM=-1
!
cat INCAR
cp INCAR INCAR.$k.DFT.shift.$ishift.vol.$fac
$VASP
cp OUTCAR OUTCAR.$k.DFT.shift.$ishift.vol.$fac

# Run initial HF calculation. This only needs a small number of basis functions.
# Since NBANDS=32 in the example here, we just set this to NBANDS.
# 
cat >INCAR <<!
ENCUT=$enc
System = $name
NBANDS=32
EDIFF  = 0.1E-06
LHFCALC=.TRUE.
AEXX=1.0
ALGO=C
SIGMA=0.001
ISYM=-1
!
cat INCAR
cp INCAR INCAR.$k.HFT.shift.$ishift.vol.$fac
$VASP
cp OUTCAR OUTCAR.$k.HFT.shift.$ishift.vol.$fac

nb=`awk <OUTCAR "/maximum number of plane-waves:/ { print \\$5 }"`

# Now re-diagonalize the HF matrix in the whole of the basis. 
# This ensures we have the whole of the virtual space available. 
# It uses the WAVECAR that was generated previously.
# $nb is now set automatically
# 
cat >INCAR <<!
ENCUT=$enc
System = $name
NBANDS=$nb
EDIFF  = 0.1E-06
LHFCALC=.TRUE.
AEXX=1.0
ALGO=sub
NELM=1
SIGMA=0.001
ISYM=-1
!
cat INCAR
cp INCAR INCAR.$k.HFTdiag.shift.$ishift.vol.$fac
$VASP
cp OUTCAR OUTCAR.$k.HFTdiag.shift.$ishift.vol.$fac


# The line below loops over different numbers of bands per k-point
# Here, we are just using one basis set NBANDS=32
# 
for nb in 32
do

rm T2CAR T1CAR FTODCAR

# This either performs the 100 MP2 calculations or does the sfTA-CCSD 
# depending on the flags set at the start of the script
# 
cat >INCAR <<!
NBANDS = $nb
ALGO=CCSD
CCMAXIT= $ccmaxit
$MP2flag
LSFACTOR=.TRUE.
PRECFOCK=N
LMAXFOCKAE=4
LHFCALC=.TRUE.
AEXX=1.0
ENCUT = $enc
ENCUTGW= $egw
ISYM=-1
!
cat INCAR
cp INCAR INCAR.TA.$x.kp.$k$k$k.$ccfilehandle.nbands.$nb.shift.$ishift.vol.$fac
$VASP
cp OUTCAR OUTCAR.TA.$x.kp.$k$k$k.$ccfilehandle.nbands.$nb.shift.$ishift.vol.$fac
cp CORRofG CORRofG.TA.$x.kp.$k$k$k.$ccfilehandle.nbands.$nb.shift.$ishift.vol.$fac
rm WAVECAR
rm T2CAR T1CAR FTODCAR
x=`expr $x + 1`
done

cd ..

done

done

cd ..

done

# Uncomment the line below to keep the WAVECAR file (these are very large)
# rm WAVECAR
