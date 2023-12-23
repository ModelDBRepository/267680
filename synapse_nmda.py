#Dainauskas JJ, Marie H, Migliore M and Saudargiene A.  
#GluN2B-NMDAR subunit contribution on synaptic plasticity: a phenomenological model for CA3-CA1 synapses
#Frontiers in Synaptic Neuroscience 2023 DOI: 10.3389/fnsyn.2023.1113957

#NMDA synapse model 

import numpy as np


class Synapse_NMDA:
    def __init__(self, dt, nmdabar = 1e-3, nmdabar_nr2cd = 1e-5):
        self.nmdabar_nr2cd = nmdabar_nr2cd
        self.Cdur = 1
        self.mg = 1
        self.dt = dt
        self.t = 0
        self.t0 = -1000

        self.synon = 0
        self.g_nr2a = 0
        self.g_nr2b = 0
        self.g_nmda = 0

        #NR2B for LTP, NR2A for LTD mainly
        self.g_nmda_LTP = 0
        self.g_nmda_LTD = 0

        self.Ron_nr2a = 0
        self.Ron_nr2b = 0
        self.Roff_nr2a = 0
        self.Roff_nr2b = 0

        self.Ron_nr2cd = 0
        self.Roff_nr2cd = 0

        self.Alpha_nr2a = 0.5
        self.Beta_nr2a = 0.024

        self.Alpha_nr2b = 0.1
        self.Beta_nr2b = 0.0075

        self.Alpha_nr2cd = 0.1
        self.Beta_nr2cd =  0.002

        self.Rinf_nr2a = self.Alpha_nr2a / (self.Alpha_nr2a + self.Beta_nr2a)
        self.Rinf_nr2b = self.Alpha_nr2b / (self.Alpha_nr2b + self.Beta_nr2b)
        self.Rtau_nr2a = 1 / (self.Alpha_nr2a + self.Beta_nr2a)
        self.Rtau_nr2b = 1 / (self.Alpha_nr2b + self.Beta_nr2b)

        self.Rinf_nr2cd = self.Alpha_nr2cd / (self.Alpha_nr2cd + self.Beta_nr2cd)
        self.Rtau_nr2cd = 1 / (self.Alpha_nr2cd + self.Beta_nr2cd)
        

        self.v = 0
        self.gNMDAbar = nmdabar
        self.gNR2Abar = 1
        self.gNR2Bbar = 1
        self.gnmda = 0
        self.Rinf = 0
        self.Rtau = 0
        self.synon = 0
        self.Ron = 0
        self.Roff = 0

        self.g_nr2cd = 0
        self.gNR2CDbar = self.nmdabar_nr2cd
        
        
        self.r0_nr2cd = 0

        self.r0_nr2a = 0
        self.r0_nr2b = 0

        self.mgblock = 0

        self.flag = 0
        self.on = 0


    def dynamics(self):
        self.Ron_nr2a += ((self.synon * self.Rinf_nr2a - self.Ron_nr2a) / self.Rtau_nr2a) * self.dt
        self.Ron_nr2b += ((self.synon * self.Rinf_nr2b - self.Ron_nr2b) / self.Rtau_nr2b) * self.dt
        self.Roff_nr2a += (-self.Beta_nr2a * self.Roff_nr2a) * self.dt
        self.Roff_nr2b += (-self.Beta_nr2b * self.Roff_nr2b) * self.dt

        self.Ron_nr2cd += ((self.synon * self.Rinf_nr2cd - self.Ron_nr2cd) / self.Rtau_nr2cd) * self.dt
        self.Roff_nr2cd += (-self.Beta_nr2cd * self.Roff_nr2cd) * self.dt



    def states(self):
        self.mgblock = 1 / (1 + np.exp(-0.062 * self.v) * (self.mg / 3.57))
        self.g_nr2a = self.mgblock * (self.Ron_nr2a + self.Roff_nr2a) * self.gNR2Abar *self.gNMDAbar
        self.g_nr2b = self.mgblock * (self.Ron_nr2b + self.Roff_nr2b) * self.gNR2Bbar *self.gNMDAbar

        self.g_nmda_base = (self.g_nr2a + self.g_nr2b)
        self.g_nmda = self.g_nmda_base 

        self.g_nmda_LTP = (0.2*self.g_nr2a + 0.8*self.g_nr2b)
        self.g_nmda_LTD = (0.8*self.g_nr2a + 0.2*self.g_nr2b)

        self.mgblock_nr2cd = 1 / (1 + np.exp(-0.062 * (0)) * (self.mg / 3.57))
        self.g_nr2cd = self.mgblock_nr2cd * (self.Ron_nr2cd + self.Roff_nr2cd) * self.gNR2CDbar
        

    def step(self, t, v, spike=False):
        self.v = v
        self.t = t
        
        self.dynamics()

        self.Glu_steps=self.Cdur/self.dt  

        if spike:
            self.netreceive()
        if self.flag > 0 and self.t >= (self.t0 + self.Glu_steps):
            self.netreceive_Glu()


        self.states()

    def netreceive(self, weight=1):
        if (self.flag == 0):
            if (not self.on):
                self.r0_nr2a = self.r0_nr2a * np.exp(-self.Beta_nr2a * (self.t - self.t0) * self.dt)
                self.r0_nr2b = self.r0_nr2b * np.exp(-self.Beta_nr2b * (self.t - self.t0) * self.dt)

                self.r0_nr2cd = self.r0_nr2cd * np.exp(-self.Beta_nr2cd * (self.t - self.t0) * self.dt)

                self.on = 1
                self.t0 = self.t
                self.synon = self.synon + weight

                self.Ron_nr2a = self.Ron_nr2a + self.r0_nr2a
                self.Roff_nr2a = self.Roff_nr2a - self.r0_nr2a
                
                self.Ron_nr2b = self.Ron_nr2b + self.r0_nr2b
                self.Roff_nr2b = self.Roff_nr2b - self.r0_nr2b    

                self.Ron_nr2cd = self.Ron_nr2cd + self.r0_nr2cd
                self.Roff_nr2cd = self.Roff_nr2cd - self.r0_nr2cd           

            self.flag = 1
            return True

    def netreceive_Glu(self, weight=1):
        if (self.flag > 0):
            self.flag = 0
            self.r0_nr2a = weight * self.Rinf_nr2a + (self.r0_nr2a - weight * self.Rinf_nr2a) * np.exp((-(self.t - self.t0) * self.dt) / self.Rtau_nr2a)
            self.r0_nr2b = weight * self.Rinf_nr2b + (self.r0_nr2b - weight * self.Rinf_nr2b) * np.exp((-(self.t - self.t0) * self.dt) / self.Rtau_nr2b)
            
            self.r0_nr2cd = weight * self.Rinf_nr2cd + (self.r0_nr2cd - weight * self.Rinf_nr2cd) * np.exp((-(self.t - self.t0) * self.dt) / self.Rtau_nr2cd)
           
            
            self.t0 = self.t
            self.synon = self.synon - weight
            self.Ron_nr2a = self.Ron_nr2a - self.r0_nr2a
            self.Ron_nr2b = self.Ron_nr2b - self.r0_nr2b
            self.Roff_nr2a = self.Roff_nr2a + self.r0_nr2a
            self.Roff_nr2b = self.Roff_nr2b + self.r0_nr2b
            self.Roff_nr2cd = self.Roff_nr2cd + self.r0_nr2cd
            self.Roff_nr2cd = self.Roff_nr2cd + self.r0_nr2cd
            self.on = 0

        return False

