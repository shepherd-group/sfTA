# Name: Tina Mihm
# Date: 01/14/2020 - modified 10/13/2020 - streamlined more for ease of use
# Description: Uses the transition structure factor data call "CORRofG" from VASP MP2 calculations to find the special twist angle for use in the CCSD correlation energy calculation


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager, FontProperties

#code below uses latex to help format figure.
#This code can be commented out and the code will still run.
#The plotting setting will go back to default in python

###
### SET UP FIGURE
###
plt.rcParams['text.usetex'] = True
###
### Fonts
###
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Computer Modern Roman'
plt.rcParams['legend.numpoints'] = 1
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 6.0
plt.rcParams['savefig.dpi'] = 100
plt.rcParams['figure.dpi'] = 200
###
### Size of the figure
###
ratio=(np.sqrt(5)-1)/2.0     # golden ratio

plt.rcParams["figure.figsize"] = 3.37, (3.37)*ratio


#################################################################################
# Data set up:                                                                  #
#   Initialize pandas data-frame with transition structure factor data from csv.#
#   Data is sorted smallest to largest first by twist angle number ("File ID")  #
#   and then by the G ("G").                                                    #
#   Data is rounded off to 10 decimals to ensure all floats are the same length #
#   and to remove any trailing "0"                                              #
#################################################################################
Data = pd.read_csv("Na32_Data.csv")

Data = Data.sort_values(by = ["Twist angle Num", "G"], ascending=True)
Data = Data.round(10)

#################################################################################
# Grouping and averaging:                                                       #
#   Data is grouped by the twist angle number and assigned to the variable      #
#    "grouped"                                                                  #
#   Data is also averaged over each unique G for each twist angle's structure   #
#   factor and assigned to the variable "grouped_average".                      #
#   This data is then printed to a new csv.                                   #
#################################################################################

grouped = Data.groupby('Twist angle Num')

grouped_average = Data.groupby(['Twist angle Num', 'G']).mean()

grouped_average.to_csv(r'SFTA-MP2_Averaged_over_G_Data.csv')

#################################################################################
#Energies:                                                                      #
#   Loops over the "grouped" data frame using the twist angle number as the     #
#   name ("n") of each of the MP2 structure factor data frames ("g")            #
#   The energy is then calculated using the equation:                           #
#                        E = sum_G S(G)*V(G)                                    #
#                                                                               #
#    The twist angle number and final accumulated energy values for each twist  #
#    angle are then gathered into a data frame                                  #
#################################################################################

E_c = 0
E_Cor = []
Name = []
for n, g in grouped:
    Name += [n] ## Gathers twist angle number that has just been looped over
    E_Cor.append((g['V_G']*g['S_G']).sum()) ## Adds the calculated twist angle energy from above to a list
TA_data = {'Twist Angle': Name, "E_c": E_Cor}
df4 = pd.DataFrame(TA_data)


#################################################################################
# Averaging:                                                                    #
#   Data is averaged over each unique G for all twist angle data to give the    #
#   averaged structure factor data. Assigned to variable "Average"              #
#   The standard error for all 100 MP2 twist angles is calculated for each data #
#   point in the averaged structure factor and assigned to variable "Error"     #
#                                                                               #
#   Structure factor data is then grouped and averaged over each unique G in    #
#   each twist angle and then re-grouped by twist angle. Assigned               #
#   variable "New_data"                                                         #
#   Structure factor data is averaged over each unique G for all twist angle    #
#   data to give the averaged structure factor data with G index for use in     #
#   finding the special twist angle. The G index use is removed to reset index  #
#   to the twist angle ("File ID") to match "New_data" index for use in finding #
#   the special twist angle in the next step. Assigned variable "Average2"      #
#################################################################################

Average = Data.groupby("G", as_index = False).mean()
print("This is the averaged structure factor data:")
print(Average)
n_twst = 100 # number of twist angles used
Error = (Data.groupby("G", as_index = False).std())/(np.sqrt(n_twst))
print("this is error in the average structure factor:")
print(Error)

New_data = Data.groupby(['Twist angle Num', 'G'], as_index = False).mean().groupby('Twist angle Num')

Average2 = Data.groupby(['Twist angle Num', 'G'], as_index=False).mean().groupby('G', as_index=False).mean()


#################################################################################
# Finding the Special Twist Angle:                                              #
#   The "New_data" is looped over with the twist angle number as the name of    #
#   the group ("n") and the structure factor data for that twist angle as the   #
#   group ("g")                                                                 #
#   the square difference between every point in each twist angle's structure   #
#   factor and the Average structure factor is calculate using the              #
#   following equation:                                                         #
#                         Diff = (S(G) - S_av(G))^2                             #
#                                                                               #
#   The differences are then added to the structure factor data for each        #
#   twist angle                                                                 #
#                                                                               #
#   The data is then looped over again, using the twist angle number as the     #
#   name of the group ("n") and the structure factor data for that twist angle  #
#   as the group ("g")                                                          #
#   the differences are then added together for each twist angle using the      #
#   following equation:                                                         #
#                         Sum = sum_G (Diff)                                    #
#                                                                               #
#   The smallest sum of the squared differences across all twist angles is      #
#   found using the new sums data frame and assigned the variable "Dif_min_G"   #
#                                                                               #
#   This variable is then used to find the python index ("SG_IN") for the twist #
#   angle number that is designated the BasisSetData twist angle. As python     #
#   indexing starts from "0", the special twist angle number, then, is          #
#   (SG_IN + 1)                                                                 #
#                                                                               #
#################################################################################

Dif_G = []

for _,g in New_data:
    for i in range(0, len(Average2)):
        Dif_G.append((g['S_G'].iloc[i]-Average2['S_G'][i])**2 ) # Gathers the differences above into a list
grouped_average['DiffSq_SG'] = Dif_G

n_d = grouped_average.reset_index() # Removes any index settings for the grouped2 data frame
N_D = n_d.groupby('Twist angle Num') # Regroups by twist angle number

Sum_Dif_G = []

for _,g in N_D:
    Sum_Dif_G.append(g['DiffSq_SG'].sum()) # Gathers the sums from above into a list


df_sums = {'Twist Angle': Name, "Sg sum Sqr diff": Sum_Dif_G}
df_sums = pd.DataFrame(df_sums)


Dif_min_G = min(Sum_Dif_G)
SG_IN = Sum_Dif_G.index(Dif_min_G)
#--------------------------------------------------------------------------------------------------------
# Set up to produce three graphs:
#  1. Just the 100 twist angle structure factors (Fig 1)
#  2. The 100 twist angle structure factors and the averaged structure factor (Fig 2)
#  3. The 100 twist angle structure factors, averaged structure factor and the BasisSetData structure factor (Fig 3)

Graph_data = Data.groupby(['Twist angle Num', 'G'], as_index = False).mean().groupby('Twist angle Num') # Sets up data to graph the 100 individual twist angle structure factors

plt.figure(1)
for n, g in Graph_data:
    plt.plot('G', "S_G",
             data = g,
             linestyle = '-',
             label = '$S(G), MP2$' if n==1 else '', # plot label only once
             marker = '', color = "#02a642",
             alpha = 1.0,
             markeredgewidth = 1.0 ,
             markersize = 5.0,
             markeredgecolor = 'black')

    plt.ylabel(r'S(G)')
    plt.xlabel(r'G')
    plt.legend(loc='center', bbox_to_anchor=(0.7, 0.5), ncol=1, handlelength = 1.0, handletextpad = 0.1, columnspacing=1.0)

#plt.savefig('SFTA-MP2_S_g_vs_G-TwstAnlgeSg.png',dpi = 300, bbox_inches = 'tight')
# Graphs saved as png as too much data for pdf to render efficiently
#-------------------------------------------------------------------------------------
plt.figure(2)
for n, g in Graph_data:

    plt.rc('text', usetex=True)
    plt.plot('G', "S_G",
             data = g,
             linestyle = '-',
             label = '$S(G), MP2$' if n==1 else '', # plot label only once
             marker = '', color = "#02a642",
             alpha = 1.0)

plt.errorbar('G', "S_G", data = Average, yerr=Error["S_G"], linestyle = '-', label = 'TA-MP2, $S(G)$', marker = 'o', color = "#2c43fc", alpha = 1.0, markeredgewidth = 1.0 , markersize = 5.0, markeredgecolor = 'black', zorder=10)
plt.ylabel(r'S(G)')

plt.xlabel(r'G')
plt.legend(loc='center', bbox_to_anchor=(0.75, 0.25), ncol=1, handlelength = 1.0, handletextpad = 0.1, columnspacing=1.0)

#plt.savefig('SFTA-MP2-SofG_vs_G-Full+Average.png',dpi = 300, bbox_inches = 'tight')

#-------------------------------------------------------------------------
plt.figure(3)
for n, g in Graph_data:

    plt.rc('text', usetex=True)
    plt.plot('G', "S_G",
             data = g,
             linestyle = '-',
             label = '$S(G), MP2$' if n==1 else '', # plot label only once
             marker = '', color = "#02a642",
             alpha = 1.0)

    if int(n) == SG_IN+1: # Adding 1 to python index to get the special twist angle number
        plt.rc('text', usetex=True)
        plt.plot('G', "S_G", data = g, linestyle = '--', label = 'BasisSetData-MP2, $S(G)$', marker = 'o', color = "#f26003", alpha = 1.0, markeredgewidth = 1.0 , markersize = 5.0, markeredgecolor = 'black', zorder = 15)

plt.errorbar('G', "S_G", data = Average, yerr=Error["S_G"], linestyle = '-', label = 'TA-MP2, $S(G)$', marker = '', color = "#2c43fc", alpha = 1.0, markeredgewidth = 1.0 , markersize = 5.0, markeredgecolor = 'black', zorder=10)

plt.ylabel(r'S(G)')

plt.xlabel(r'G')
plt.legend(loc='center', bbox_to_anchor=(0.70, 0.35), ncol=1, handlelength = 1.0, handletextpad = 0.1, columnspacing=1.0)

plt.savefig('SFTA-MP2-SofG_vs_G-Full+Average_With_specialTwst.png',dpi = 300, bbox_inches = 'tight')

#-------------------------------------------------

#minimum diff with index for special twist angle

Dif_min_G = min(Sum_Dif_G)
print('-------------------------------------------------------------------')
print("S_G special twist angle (min diff, BasisSetData twist angle python index)", Dif_min_G, Sum_Dif_G.index(Dif_min_G))
print('-------------------------------------------------------------------')
#------------------------------------
