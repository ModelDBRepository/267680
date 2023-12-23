#Dainauskas JJ, Marie H, Migliore M and Saudargiene A.  
#GluN2B-NMDAR subunit contribution on synaptic plasticity: a phenomenological model for CA3-CA1 synapses
#Frontiers in Synaptic Neuroscience 2023 DOI: 10.3389/fnsyn.2023.1113957

#AMPA synapse model 

import numpy as np

class Synapse_AMPA:
    def __init__(self, dt, ampabar = 1):
        self.Cdur = 1
        self.gampa_bar = ampabar
        self.weight_ampa = 0.1
        self.Alpha_ampa	= 1.1
        self.Beta_ampa	= 0.19
        self.Rinf_ampa = self.Alpha_ampa / (self.Alpha_ampa + self.Beta_ampa)
        self.Rtau_ampa = 1 / (self.Alpha_ampa + self.Beta_ampa)
        self.duration_Glu=1
        self.E_AMPA = 0
        
        self.r0_ampa = 0
        self.Ron_ampa = 0
        self.Roff_ampa = 0
        self.synon_ampa = 0
        self.nspike_ampa = 0
        self.on_ampa = 0
        self.g_ampa = 0

        self.dt = dt
        self.t = 0
        self.t0 = -1000
        self.flag = 0


    def dynamics(self):
        self.delta_Ron_ampa = (self.synon_ampa * self.Rinf_ampa - self.Ron_ampa) / self.Rtau_ampa
        self.delta_Roff_ampa = -self.Beta_ampa * self.Roff_ampa
        self.Ron_ampa = self.Ron_ampa + self.delta_Ron_ampa * self.dt
        self.Roff_ampa = self.Roff_ampa + self.delta_Roff_ampa * self.dt

    def states(self):
        self.g_ampa = (self.Ron_ampa + self.Roff_ampa) * self.gampa_bar


    def step(self, t, v, spike=False):
        self.v = v
        self.t = t
        
        self.dynamics()

        self.Glu_steps = self.Cdur / self.dt  

        if spike:
            self.netreceive()
        if self.flag > 0 and self.t >= (self.t0 + self.Glu_steps):
            self.netreceive_Glu()

        self.states()

    def netreceive(self, weight=1):
        if (self.flag == 0):
            if (not self.on_ampa):
                self.nspike_ampa = self.nspike_ampa+1
                self.r0_ampa = self.r0_ampa * np.exp(-self.Beta_ampa*(self.t - self.t0))
                self.Ron_ampa = self.Ron_ampa + self.r0_ampa
                self.Roff_ampa = self.Roff_ampa - self.r0_ampa
                self.on_ampa = 1
                self.synon_ampa = self.synon_ampa + self.weight_ampa
                self.t0 = self.t
            
            self.flag = 1

    def netreceive_Glu(self, weight=1):
        if (self.flag > 0):
            self.flag = 0
            self.r0_ampa = self.weight_ampa * self.Rinf_ampa + (self.r0_ampa - self.weight_ampa * self.Rinf_ampa) * np.exp(-(self.t - self.t0)/self.Rtau_ampa)
            self.synon_ampa = self.synon_ampa - self.weight_ampa
            self.Ron_ampa = self.Ron_ampa - self.r0_ampa
            self.Roff_ampa = self.Roff_ampa + self.r0_ampa
            self.on_ampa = 0
            
            self.t0 = self.t
        

