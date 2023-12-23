#Dainauskas JJ, Marie H, Migliore M and Saudargiene A.  
#GluN2B-NMDAR subunit contribution on synaptic plasticity: a phenomenological model for CA3-CA1 synapses
#Frontiers in Synaptic Neuroscience 2023 DOI: 10.3389/fnsyn.2023.1113957.

#Model of synaptic plasticity

import numpy as np
from neuron_model import CA1_NeuronModel
from synapse_nmda import Synapse_NMDA
from synapse_ampa import Synapse_AMPA
from multiprocessing import Pool

class PylediPlasticity:
    
    def __init__(self, nmdabar=1e-3, ampabar=0.5, ampabar_multiplier=1, save_last_record_only=False):
        self.save_last_record_only = save_last_record_only
        self.set_records_list()
        self.init_parameters()
        self.init_records()
        self.nmdabar = nmdabar
        self.ampabar = ampabar * ampabar_multiplier
        self.stdp_range = list(range(-100, 110, 10))

        self.neuron = CA1_NeuronModel(self.dt, save_last_record_only=save_last_record_only)
        self.synapse_nmda = Synapse_NMDA(self.dt, nmdabar=self.nmdabar)
        self.synapse_ampa = Synapse_AMPA(self.dt, ampabar=self.ampabar)
        
    def init_parameters(self):
        self.t = 0.0
        self.dt = 0.1 # ms

        self.Ip0_ampl = 20
        self.Ip0_ampl_rest = -0.5

        #Membrane parameters
        self.Vrest = -70    #mV
        self.v = self.Vrest
        self.E_AMPA = 0
        self.E_NMDA = 0

        # AMPA 
        self.g_ampa = 0.0
        self.synInputs = 0.0

        self.I_NMDA = 0.0
        self.I_AMPA = 0.0

        # NMDA
        self.g_nmda = 0.0
        self.g_nmda_LTP = 0
        self.g_nmda_LTD = 0

        self.g_nr2a = 0.0
        self.g_nr2b = 0.0
     
        # NMDA traces 
        self.g_nmda_tau1 = 20
        self.g_nmda_tau2 = 1000
        self.g_nmda_trace1 = 0.0
        self.g_nmda_trace2 = 0.0
            
        self.k_gnmda_scaling=1        
        
        # Synaptic plasticity model
        self.v_trace1_threshed1 = 0.0
        self.v_trace2_threshed2 = 0.0
    
        self.pre_dirac = 0.0
        self.dirac_trace = 0.0

        self.A_dirac = 1

        self.hill_ltp_mid = 11e-5
        self.hill_ltd_mid = 9e-5
       
        self.hill_ltp_coef = 4
        self.hill_ltd_coef = 2
       
        self.moving_threshold_hill_ltp = 0.0
        self.moving_threshold_hill_ltp_multiplier_from_ltd = 10
        self.moving_threshold_hill_ltp_tau = 100

        self.moving_threshold_hill_ltd = 0.0
        self.moving_threshold_hill_ltd_multiplier_from_ltp = 1000   
        self.moving_threshold_hill_ltd_tau = 100

        self.A_ltp = 1 
        self.A_ltd = 100

        self.v_trace1_tau = 10
        self.v_trace2_tau = 10

        self.v_trace_thresh1 = -65.0    
        self.v_trace_thresh2 = -67.0    

        self.dirac_trace_tau = 15

        self.wmin = 0.4
        self.wmax = 2.0

        self.weight = 1.0
        self.customweight = 1.0

        self.w_ltd = 0.0
        self.w_ltp = 0.0

        self.w_change = 0.0
        self.ltp_part = 0.0
        self.ltd_part = 0.0
        

    #-------------------------------------------------------------------------
    # Record
    #-------------------------------------------------------------------------    
    def set_records_list(self):
        self.records_list = [
            "g_ampa",
            "g_nmda",
            "g_nmda_trace1",
            "g_nmda_trace2",
            "t",
            "v",
            "v_soma",
            "v_trace1_threshed1",
            "v_trace2_threshed2",
            "w_ltd",
            "w_ltp",
            "ltd_part",
            "ltp_part",
            "pre_dirac",
            "dirac_trace",
            "weight",
            "hilleq_ltp",
            "hilleq_ltd",
            "g_nr2a",
            "g_nr2b",
            "moving_threshold_hill_ltd",
            "moving_threshold_hill_ltp",
        ]

    def init_records(self):
        self.records = {}
        if self.save_last_record_only:
            for record_name in self.records_list:
                self.records[record_name] = []
                self.records[record_name].append(0)
        else:
            for record_name in self.records_list:
                self.records[record_name] = []
        
    def update_records(self):
        if self.save_last_record_only:
            for record_name in self.records_list:
                self.records[record_name][-1] = getattr(self, record_name)
        else:
            for record_name in self.records_list:
                self.records[record_name].append(getattr(self, record_name))
        
    def step(self, i, syn_input=False, soma_input=False):
        Ip0_ampl = self.Ip0_ampl_rest
        if soma_input:
            Ip0_ampl = self.Ip0_ampl
        self.neuron.step(i, self.synInputs, self.I_NMDA, Ip0_ampl)
        self.synapse_ampa.step(i, self.v, syn_input)
        self.synapse_nmda.step(i, self.v, syn_input)
        self.g_ampa = self.synapse_ampa.g_ampa * self.weight * self.customweight
        self.g_nmda = self.synapse_nmda.g_nmda
        self.g_nr2a = self.synapse_nmda.g_nr2a
        self.g_nr2b = self.synapse_nmda.g_nr2b
        self.v = self.neuron.Vdend
        self.v_soma = self.neuron.Vsoma
        
        #NMDA for LTP and LTD
        self.g_nmda_LTP = self.synapse_nmda.g_nmda_LTP
        self.g_nmda_LTD = self.synapse_nmda.g_nmda_LTD

        self.pre_dirac = 0.0
        if syn_input:
            self.net_receive()

        self.dynamics()
        self.update()
        self.t += self.dt
        self.update_records()


    #--------------------------------------------------------------------------------
    #Synaptic plasticity model
    #--------------------------------------------------------------------------------

    def dynamics(self):
        self.v_trace1_threshed1 += ((max(0, (self.v - self.v_trace_thresh1)) - self.v_trace1_threshed1) / self.v_trace1_tau) * self.dt
        self.v_trace2_threshed2 += ((max(0, (self.v - self.v_trace_thresh2)) - self.v_trace2_threshed2) / self.v_trace2_tau) * self.dt
        
        self.dirac_trace += ((self.pre_dirac - self.dirac_trace) / self.dirac_trace_tau) * self.dt

        #--------------------
        #NMDA traces
        self.g_nmda_trace1 += (((self.g_nmda_LTP) - self.g_nmda_trace1) / self.g_nmda_tau1) * self.dt
        self.g_nmda_trace2 += (((self.g_nmda_LTD) - self.g_nmda_trace2) / self.g_nmda_tau2) * self.dt

    def net_receive(self):
        self.pre_dirac = self.A_dirac

    def update(self):        
        self.I_AMPA = self.g_ampa * (self.v - self.E_AMPA)
        self.I_NMDA = self.g_nmda * (self.v - self.E_NMDA) * self.nmdabar
        self.synInputs = self.I_AMPA + self.I_NMDA

        #LTP
        hill_ltp = (self.k_gnmda_scaling * self.g_nmda_trace1) ** self.hill_ltp_coef
        self.hilleq_ltp = hill_ltp / ((self.hill_ltp_mid) ** self.hill_ltp_coef + hill_ltp) - self.moving_threshold_hill_ltp
        if self.hilleq_ltp < 0:
            self.hilleq_ltp = 0
        self.ltp_part = self.A_ltp * self.hilleq_ltp * self.v_trace1_threshed1

        #LTD
        hill_ltd = (self.k_gnmda_scaling * self.g_nmda_trace2) ** self.hill_ltd_coef
        self.hilleq_ltd = hill_ltd / ((self.hill_ltd_mid) ** self.hill_ltd_coef + hill_ltd) - self.moving_threshold_hill_ltd
        if self.hilleq_ltd < 0:
            self.hilleq_ltd = 0
        self.ltd_part = self.A_ltd * self.hilleq_ltd  * self.v_trace2_threshed2 * self.dirac_trace

        #LTP threshold 
        self.moving_threshold_hill_ltp += ((-self.moving_threshold_hill_ltp + self.hilleq_ltd * self.moving_threshold_hill_ltp_multiplier_from_ltd) / self.moving_threshold_hill_ltp_tau) * self.dt
        #LTD threshold
        self.moving_threshold_hill_ltd += ((-self.moving_threshold_hill_ltd + self.hilleq_ltp * self.moving_threshold_hill_ltd_multiplier_from_ltp) / self.moving_threshold_hill_ltd_tau) * self.dt
        
        ltpmult = (self.wmax - self.weight)
        ltdmult = (self.weight - self.wmin)

        self.w_change = (self.ltp_part * ltpmult - self.ltd_part * ltdmult)

        self.w_ltp += self.ltp_part * ltpmult * self.dt
        self.w_ltd += self.ltd_part * ltdmult * self.dt
        
        self.weight += self.w_change * self.dt
        
        if self.weight < 0:
            self.weight = 0
    
    def get_record_data(self):
        return self.records

    def run_stdp_tests_static(pairings = 2, frequency = 0.5, nmdabarmult = 1, pre_spikes = 1, post_spikes = 1, synparams = {}, stdp_range=range(-100, 110, 10), first_post=False, save_last_record_only=False):
        taskrange = [(pairings, frequency, stdp_dt, nmdabarmult, pre_spikes, post_spikes, synparams, first_post, save_last_record_only) for stdp_dt in stdp_range]
        with Pool(len(taskrange)) as p:
            experiments = p.map(PylediPlasticity.run_single_stdp_static, taskrange)
        return experiments
        
    def run_single_stdp_static(input_tuple):
        (pairings, frequency, stdp_dt, nmdabarmult, pre_spikes, post_spikes, synparams, first_post, save_last_record_only) = input_tuple
        model = PylediPlasticity(save_last_record_only=save_last_record_only)

        for synparam in synparams:
            setattr(model, synparam, synparams[synparam])
            
        dt = model.dt
        pre_pair_interval = 10
        pre_pair_interval_dt = pre_pair_interval / dt
        post_pair_interval = 10
        post_pair_interval_dt = post_pair_interval / dt

        interval = int(1000/frequency)

        start = 100
        stop = interval * (pairings-1) + start + 300

        dt_int_pair = int((interval * pairings)/dt)
        dt_start_dend = int((start + 2)/dt)
        dt_start_soma = int(start/dt)
        dt_stop = int(stop/dt)    

        dt_interval = int(interval/dt)
        dt_stdp_dt = int(stdp_dt/model.dt)

        syn_inputs = []
        if first_post:
            pre_delay = 0
        else:
            pre_delay = post_pair_interval_dt * (post_spikes-1)

        for i in range(pre_spikes):
            syn_inputs += list(np.arange(dt_start_dend + pre_pair_interval_dt * i - dt_stdp_dt + pre_delay, dt_start_dend + dt_int_pair - dt_stdp_dt, dt_interval))


        soma_delta = -1.0 / dt
        soma_inputs = []
        for i in range(post_spikes):
            soma_inputs += list(np.arange(dt_start_soma + soma_delta + post_pair_interval_dt * i, dt_start_soma + soma_delta + dt_int_pair, dt_interval))

        soma_input_period_ms = 5

        soma_input_period = 0
        for i in range(dt_stop):
            syn_input_on = False
            if  soma_input_period > 0:
                soma_input_period -= 1
            if (i in syn_inputs):
                syn_input_on = True
            if (i in soma_inputs):
                soma_input_period = soma_input_period_ms / model.dt
            model.step(i, syn_input_on, soma_input_period > 0)
        
        return model.get_record_data()

    def run_freq_tests(self, frequency=5.0, pairings=2, soma_on=False, stdp_dt=10, start=300, afterstop=100, old_i = 0):
        interval = int(1000/frequency)
        start = 300
        afterstop = 100
        stop = interval * (pairings-1) + start + afterstop

        
        dt_interval = int(interval/self.dt)
        dt_start = int(start/self.dt)
        dt_stop = int(stop/self.dt)

        syn_inputs = np.arange(dt_start, dt_start + (dt_interval * pairings), dt_interval)
        soma_inputs = []
        if soma_on:
            soma_inputs = np.arange(dt_start - int(5 / self.dt) + int(stdp_dt/self.dt), dt_start - int(5 / self.dt) + int(stdp_dt/self.dt) + (dt_interval * pairings), dt_interval)

        soma_input_period_ms = 5

        soma_input_period = 0
        for i in range(dt_stop):
            syn_input_on = False
            if  soma_input_period > 0:
                soma_input_period -= 1
            if (i in syn_inputs):
                syn_input_on = True
            if (i in soma_inputs):
                soma_input_period = soma_input_period_ms / self.dt
            self.step(i+old_i, syn_input_on, soma_input_period > 0)
        self.run_data = self.get_record_data()
        return dt_stop
