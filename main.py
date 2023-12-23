#Dainauskas JJ, Marie H, Migliore M and Saudargiene A.  
#GluN2B-NMDAR subunit contribution on synaptic plasticity: a phenomenological model for CA3-CA1 synapses
#Frontiers in Synaptic Neuroscience 2023 DOI: 10.3389/fnsyn.2023.1113957.
#ausra.saudargiene@lsmu.lt

#Main code 
#Reproduces Fig2, Fig3A, Fig3B, Fig3C, Fig3D, Fig4A, Fig4B, Fig5 

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from plotting import *
from pathlib import Path
from plasticity_model import PylediPlasticity
import os


def run_freq_single_processing(data, ampabar_multiplier=1):
    frequency, pairings = data
    model = PylediPlasticity(ampabar_multiplier=ampabar_multiplier)
    model.run_freq_tests(frequency, pairings, soma_on=False, stdp_dt=10, start=300, afterstop=100)
    return (model, frequency, pairings)


#----------------------------------------------------------
# Fig2: STDP traces LTP and LTD
#----------------------------------------------------------
pairings_STDP_2Columns=4
frequency_STDP_2Columns=5

def Fig2():
    print('Running Fig2 ...')
    dtdict = dict([(i, x) for (x, i) in enumerate(range(-100, 110, 10))])
    data = PylediPlasticity.run_stdp_tests_static(pairings = pairings_STDP_2Columns, frequency = frequency_STDP_2Columns, post_spikes = 2)
    DT = 10
    datas = [data[dtdict[DT]], data[dtdict[-DT]]]
    plot_2columns(datas, filename=f"Fig2.png")

#----------------------------------------------------------
# Fig3A: STDP LTD LTP LTD
#----------------------------------------------------------
pairings_STDP_60pairs=60
frequency_STDP_60pairs=5 

def Fig3A():
    print('Running Fig3A ...')
    results_multiple = []
    for item in [0]:
        synparams = {}
        results = PylediPlasticity.run_stdp_tests_static(pairings = pairings_STDP_60pairs, frequency = frequency_STDP_60pairs, post_spikes = 2, synparams = synparams, save_last_record_only=True)
        results_multiple.append(results)
    plot_stdp_full_60pairs(results_multiple, filename="figures/Fig3A.png")
    
#----------------------------------------------------------
# Fig3B: STDP LTP
#----------------------------------------------------------
pairings_STDP_5pairs=5
frequency_STDP_5pairs=5

def Fig3B():
    print('Running Fig3B ...')
    results_multiple = []
    for item in [0]:
        synparams = {}
        results = PylediPlasticity.run_stdp_tests_static(pairings = pairings_STDP_5pairs, frequency = frequency_STDP_5pairs, post_spikes = 2, synparams = synparams, save_last_record_only=True)
        results_multiple.append(results)
    plot_stdp_full_5_30pairs(results_multiple, filename="figures/Fig3B.png")
    
#----------------------------------------------------------
# Fig3C: STDP LTD
#----------------------------------------------------------
pairings_STDP_30pairs=30
frequency_STDP_30pairs=1

def Fig3C():
    print('Running Fig3C ...')
    results_multiple = []
    for item in [0]:
        synparams = {}
        results = PylediPlasticity.run_stdp_tests_static(pairings = pairings_STDP_30pairs, frequency = frequency_STDP_30pairs, post_spikes = 2, synparams = synparams, save_last_record_only=True)
        results_multiple.append(results)
    plot_stdp_full_5_30pairs(results_multiple, filename="figures/Fig3C.png")

#----------------------------------------------------------
# Fig3D: STDP pre - single post
#----------------------------------------------------------
pairings_STDP=60
frequency_STDP=5
def Fig3D():
    print('Running Fig3D ...')
    results = PylediPlasticity.run_stdp_tests_static(pairings = pairings_STDP, frequency = frequency_STDP, save_last_record_only=True)
    plot_stdp_full_singlePost(results, filename="figures/Fig3D.png")

#----------------------------------------------------------
# Fig4A: Frequency of pre-post
#----------------------------------------------------------
pairings_frequency = 20
    
def Fig4A():
    print("Running Fig4A ...")
    results_multiple = []
    frequencies = [1, 5, 10, 20, 30, 40, 50 ]
    for freq in frequencies:
        synparams = {}
        results = PylediPlasticity.run_stdp_tests_static(pairings = pairings_frequency, frequency = freq, post_spikes = 1, synparams = synparams, stdp_range=[10], first_post=True, save_last_record_only=True)
        results_multiple.append(results[0]['weight'][-1])
    plot_frequency_pre_post(frequencies, results_multiple, filename="figures/Fig4A.png")

#----------------------------------------------------------
# Fig4B: Number of post spikes 1-4
#----------------------------------------------------------
frequency_SpikeCount = 5
pairings_SpikeCount = 30

def Fig4B():
    print("Running Fig4B ...")
    results_multiple = []
    postspikes = [1, 2, 3, 4]
    for post_spike in postspikes:
        synparams = {}
        results = PylediPlasticity.run_stdp_tests_static(pairings = pairings_SpikeCount, frequency = frequency_SpikeCount, post_spikes = post_spike, synparams = synparams, stdp_range=[10], first_post=True, save_last_record_only=True)
        results_multiple.append(results[0]['weight'][-1])
    plot_spike_count_barplot(postspikes, results_multiple, filename="figures/Fig4B.png")

#----------------------------------------------------------
# Fig5: Frequency dependent LTP, LTP and LTD
#----------------------------------------------------------
def Fig5():
    print('Running Fig5 ...')
    (model1, _, _) = run_freq_single_processing((100, 100), ampabar_multiplier=0.5)
    (model2, _, _) = run_freq_single_processing((100, 100))
    (model3, _, _) = run_freq_single_processing((1, 3))
    plot_3columns([model1.run_data, model2.run_data, model3.run_data], filename=f"Fig5.png")

#----------------------------------------------------------

if __name__ == '__main__':
    
    FIGURES_DIR = "figures"
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)

    #Plot figures 
    Fig2()
    Fig3A()
    Fig3B()
    Fig3C()
    Fig3D()
    Fig4A() 
    Fig4B() 
    Fig5() 

    print("Done")