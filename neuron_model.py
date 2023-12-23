#Dainauskas JJ, Marie H, Migliore M and Saudargiene A.  
#GluN2B-NMDAR subunit contribution on synaptic plasticity: a phenomenological model for CA3-CA1 synapses
#Frontiers in Synaptic Neuroscience 2023 DOI: 10.3389/fnsyn.2023.1113957.

#Neuron model 
#Two compartmental Pinsky-Rinzel (1994), Ferguson-Campbell (2009) 

import numpy as np

class CA1_NeuronModel():
    def __init__(self, dt=0.01, save_last_record_only=True):
        self.save_last_record_only = save_last_record_only
        self.dt=dt

        self.VL=-65
        self.Vsoma=-60
        self.Vdend=-60

        self.gLs=0.1  
        self.gLd=0.1  
        self.gNa=30  
        self.gKdr=17 
        self.gCa_dend=5 
        self.gKahp_dend=0.8  
        self.gKC_dend=5 
        self.VNa=60  
        self.VCa=80  
        self.VK=-75  
        self.Vsyn=0  
        self.gc=1.5 
        self.pp=0.5  
        self.Cm=3  
        self.alphac=2 
        self.betac=0.1

        self.gCa_soma = 6
        self.gKahp_soma = 0.8
        self.gKC_soma = 15

        self.hs = 0.999  
        self.ns = 0.001 
        self.qd = 0.010 
        self.cd = 0.007 
        self.sd = 0.009 
        self.Cad_dend = 0.2 

        self.ss = 0.009 
        self.qs = 0.010
        self.cs_soma = 0.007 

        self.ICad = self.gCa_dend * self.sd * self.sd * (self.Vdend - self.VCa)

        self.Cad_soma = 0.2
        self.Cad_dend = 0.2

        self.Ileakage_soma = 0.5803 
        self.INa_soma = 0.0662
        self.IKdr_soma = -0.0530
        self.Icoupl_soma = 0.4072

        self.Ileakage_dend = 0.5706
        self.ICad_dend = -0.1062
        self.IAHP_dend = -0.0564 
        self.IKC_dend = -6.4859e-04
        self.Icoupl_dend = -0.4072

        self.Vsoma= -69
        self.Vdend= -69

        self.init_arrays()

    def init_arrays(self):
        self.t_plot = []
        self.Vsoma_plot = []
        self.Vdend_plot = []
        
        self.IKdr_soma_plot = []
        self.INa_soma_plot = []


        self.ns_plot = []

        self.ICad_dend_plot = []
        self.Icoupl_dend_plot = []

        self.Iext_plot = []
        
    def heaviside(self, x):
        if x < 0:
            return 0
        if x == 0:
            return 0.5
        else:
            return 1

    def run(self, n):
        for i in range(n):
            self.step(i, 0)

    def step(self, i, isyn, inmda, Ip0_ampl=0):
        
        self.t = self.dt * i
        
        self.Ip0 = Ip0_ampl
        self.I = self.Ip0
        
        # particles
        self.alpha_hs = 0.128 * np.exp((-43.0-self.Vsoma) / 18.0) 
        self.beta_hs = 4.0 / (1.0 + np.exp((-20.0-self.Vsoma) / 5.0))


        self.alpha_ms = 0.32 * (-46.9-self.Vsoma) / (np.exp((-46.9-self.Vsoma) / 4.0)-1.0)
        self.beta_ms = 0.28 * (self.Vsoma + 19.9) / (np.exp((self.Vsoma + 19.9) / 5.0)-1.0)
        self.Minfs = self.alpha_ms / (self.alpha_ms + self.beta_ms) 


        self.alpha_ns = 0.016 * (-24.9-self.Vsoma) / (np.exp((-24.9-self.Vsoma) / 5.0)-1.0)
        self.beta_ns = 0.25 * np.exp(-1.0-0.025 * self.Vsoma) 


        self.alpha_qd = min(0.00002 * self.Cad_dend,0.01) 
        self.beta_qd = 0.001 


        self.alpha_cd = (1.0 - self.heaviside(self.Vdend + 10.0)) * np.exp((self.Vdend + 50.0) / 11-(self.Vdend + 53.5) / 27) / 18.975 + self.heaviside(self.Vdend + 10.0) * 2.0 * np.exp((-53.5-self.Vdend) / 27.0) 
        self.beta_cd = (1.0 - self.heaviside(self.Vdend + 10.0)) * (2.0 * np.exp((-53.5-self.Vdend) / 27.0)-self.alpha_cd)


        self.alpha_sd = 1.6 / (1.0 + np.exp(-0.072 * (self.Vdend-5.0))) 
        self.beta_sd = 0.02 * (self.Vdend + 8.9) / (np.exp((self.Vdend + 8.9) / 5.0)-1.0)


        self.chid_dend = min(self.Cad_dend / 250.0,1.0)



        self.alpha_ss = 1.6 / (1.0 + np.exp(-0.072 * (self.Vsoma-5.0))) 
        self.beta_ss = 0.02 * (self.Vsoma + 8.9) / (np.exp((self.Vsoma + 8.9) / 5.0)-1.0)

        self.alpha_qs = min(0.00002 * self.Cad_soma,0.01) 
        self.beta_qs = 0.001 

        self.chid_soma = min(self.Cad_soma / 250.0,1.0)
        self.alpha_cs = (1.0 - self.heaviside(self.Vsoma + 10.0)) * np.exp((self.Vsoma + 50.0) / 11-(self.Vsoma + 53.5) / 27) / 18.975 + self.heaviside(self.Vsoma + 10.0) * 2.0 * np.exp((-53.5-self.Vsoma) / 27.0) 
        self.beta_cs = (1.0 - self.heaviside(self.Vsoma + 10.0)) * (2.0 * np.exp((-53.5-self.Vsoma) / 27.0)-self.alpha_cs)


        
        self.Ileakage_soma = -self.gLs * (self.Vsoma-self.VL)
        self.INa_soma = -self.gNa * (self.Minfs**2) * self.hs * (self.Vsoma-self.VNa)
        self.IKdr_soma = -self.gKdr * self.ns * (self.Vsoma-self.VK)
        self.Icoupl_soma = (self.gc / self.pp) * (self.Vdend-self.Vsoma)
        self.Iext_soma = self.Ip0 / self.pp
        self.ICad_soma = self.gCa_soma * self.sd * self.sd * (self.Vsoma-self.VCa)
        self.IAHP_soma = -self.gKahp_soma * self.qs * (self.Vsoma-self.VK)
        self.IKC_soma = -self.gKC_soma * self.cs_soma * self.chid_soma * (self.Vsoma-self.VK) 




        self.Ileakage_dend = -self.gLd * (self.Vdend-self.VL)
        self.ICad_dend = self.gCa_dend * self.sd * self.sd * (self.Vdend-self.VCa) 
        self.IAHP_dend = -self.gKahp_dend * self.qd * (self.Vdend-self.VK) 
        self.IKC_dend = -self.gKC_dend * self.cd * self.chid_dend * (self.Vdend-self.VK) 
        self.Icoupl_dend = (self.gc * (self.Vsoma-self.Vdend)) / (1.0-self.pp) 




        self.delta_Vsoma = (self.Ileakage_soma + self.INa_soma + self.IKdr_soma + self.Icoupl_soma + self.Iext_soma + self.ICad_soma + self.IAHP_soma + self.IKC_soma) / self.Cm
        self.delta_Vdend = (self.Ileakage_dend + self.ICad_dend + self.IAHP_dend + self.IKC_dend + self.Icoupl_dend - (isyn / (1.0 - self.pp))) / self.Cm

        self.delta_hs = self.alpha_hs-(self.alpha_hs + self.beta_hs) * self.hs 
        self.delta_ns = self.alpha_ns-(self.alpha_ns + self.beta_ns) * self.ns 
        self.delta_qd = self.alpha_qd-(self.alpha_qd + self.beta_qd) * self.qd 
        self.delta_cd = self.alpha_cd-(self.alpha_cd + self.beta_cd) * self.cd 
        self.delta_sd = self.alpha_sd-(self.alpha_sd + self.beta_sd) * self.sd 
        self.delta_ss = self.alpha_ss-(self.alpha_ss + self.beta_ss) * self.ss 
        self.delta_qs = self.alpha_qs-(self.alpha_qs + self.beta_qs) * self.qs 
        self.delta_cs = self.alpha_cs-(self.alpha_cs + self.beta_cs) * self.cs_soma 


        self.delta_Cad_dend = -0.13 * (self.ICad_dend + inmda * 0.06) - 0.075 * self.Cad_dend
        self.delta_Cad_soma = -0.13 * self.ICad_soma-0.075 * self.Cad_soma


        self.Vsoma = self.Vsoma + self.delta_Vsoma * self.dt
        self.Vdend = self.Vdend + self.delta_Vdend * self.dt

        self.hs = self.hs + self.delta_hs * self.dt 
        self.ns = self.ns + self.delta_ns * self.dt 
        self.qd = self.qd + self.delta_qd * self.dt 
        self.cd = self.cd + self.delta_cd * self.dt 
        self.sd = self.sd + self.delta_sd * self.dt 


        self.ss = self.ss + self.delta_ss * self.dt 
        self.qs = self.qs + self.delta_qs * self.dt 
        self.cs_soma = self.cs_soma + self.delta_cs * self.dt 

        self.Cad_soma = self.Cad_soma + self.delta_Cad_soma * self.dt
        self.Cad_dend = self.Cad_dend + self.delta_Cad_dend * self.dt

        if not self.save_last_record_only:
            self.save_records()

    def save_records(self):
        self.t_plot.append(self.t)
        self.IKdr_soma_plot.append(self.IKdr_soma)
        self.INa_soma_plot.append(self.INa_soma)

        self.Vsoma_plot.append(self.Vsoma)
        self.Vdend_plot.append(self.Vdend)

        self.ns_plot.append(self.ns)

        self.ICad_dend_plot.append(self.ICad_dend)
        self.Icoupl_dend_plot.append(self.Icoupl_dend)

        self.Iext_plot.append(self.Ip0)
        