__all__ = ['SEIRTIODE']

from typing_extensions import Self
import numpy as np
from model import modCModel
import yaml
import logging

log = logging.getLogger(__name__)

class SEIRTIODE:
    name = "SEIR-TI ODE"
    # esto lo puedo incluir en el yaml
    states_per_age = ['S_','Eu_','Ed_','Ipu_','Ipd_','Iau_','Iad_','Isu_','Isd_','R_']    
    inter_compart = ['Ipi_','Iai_','Isi_']
    states_per_age = states_per_age + inter_compart

    def set_parameters(self, **params):
        """
        Set the model parameters
        """
        for k, v in params.items():
            setattr(self, k, v)
        self.Ca = np.array(self.Ca)
        self.Cs = np.array(self.Cs)
        self.Cp = np.array(self.Cp)
        self.states = [s + k for k in self.age_groups for s in self.states_per_age]

    def initial_conditions(self, **initial):
                
        y0 = np.zeros(len(self.states))
        for k, v in initial.items():
            y0[self.states.index(k)] = v
        return y0


    def couplings(self):
       # Now add the coupling
        coupling = []
        for i,k in enumerate(self.age_groups):            
            ### State progression ###           
            # from S to Eu
            for j,k_ in enumerate(self.age_groups): 
                # from undiagnosed individuals           
                coupling.append((f'S_{k}*Ipu_{k_}:S_{k}=>Eu_{k}', self.q*self.Cp[i,j]**(-1), f'beta_pu({i},{j})'))
                coupling.append((f'S_{k}*Iau_{k_}:S_{k}=>Eu_{k}', self.q*self.Ca[i,j]**(-1), f'beta_au({i},{j})'))
                coupling.append((f'S_{k}*Isu_{k_}:S_{k}=>Eu_{k}', self.q*self.Cs[i,j]**(-1), f'beta_su({i},{j})'))
                # from diagnosed non-adherent individuals
                coupling.append((f'S_{k}*Ipd_{k_}:S_{k}=>Eu_{k}', self.n*self.q*self.Cp[i,j]**(-1), f'beta_pd({i},{j})'))
                coupling.append((f'S_{k}*Iad_{k_}:S_{k}=>Eu_{k}', self.n*self.q*self.Ca[i,j]**(-1), f'beta_ad({i},{j})'))
                coupling.append((f'S_{k}*Isd_{k_}:S_{k}=>Eu_{k}', self.n*self.q*self.Cs[i,j]**(-1), f'beta_sd({i},{j})'))

            # from Eu to Ipu
            coupling.append((f'Eu_{k}:Eu_{k}=>Ipu_{k}', self.gamma, f'gamma_u_{k}'))
            
            # from Ed to Ipd
            coupling.append((f'Ed_{k}:Ed_{k}=>Ipd_{k}', self.gamma, f'gamma_d_{k}'))
            
            # from Ipu to Iau
            coupling.append((f'Ipu_{k}:Ipu_{k}=>Iau_{k}', self.p*self.theta, f'p_u_{k}'))
            
            # from Ipu to Isu
            coupling.append((f'Ipu_{k}:Ipu_{k}=>Isu_{k}', (1-self.p)*self.theta, f'q_u_{k}'))

            # from Ipd to Iad
            coupling.append((f'Ipd_{k}:Ipd_{k}=>Iad_{k}', self.p*self.theta, f'p_d_{k}'))
            
            # from Ipd to Isd
            coupling.append((f'Ipd_{k}:Ipd_{k}=>Isd_{k}', (1-self.p)*self.theta, f'q_d_{k}'))
            
            # from Iau to R
            coupling.append((f'Iau_{k}:Iau_{k}=>R_{k}', self.delta1, f'delta1_u_{k}'))
            
            # from Iad to R
            coupling.append((f'Iad_{k}:Iad_{k}=>R_{k}', self.delta1, f'delta1_d_{k}'))
            
            # from Isu to R
            coupling.append((f'Isu_{k}:Isu_{k}=>R_{k}', self.delta2, f'delta2_u_{k}'))
            
            # from Isd to R
            coupling.append((f'Isd_{k}:Isd_{k}=>R_{k}', self.delta2, f'delta2_d_{k}'))

            ### Testing (all forms) ###            

            # from Eu to Ed, random + universal testing
            coupling.append((f'Eu_{k}:Eu_{k}=>Ed_{k}', self.tau_r + self.tau_u, f'D_E_{k}'))

            # from Ipu to Ipd, random + universal testing using intermetiate compartment to allow contact tracing
            coupling.append((f'Ipu_{k}:Ipu_{k}=>Ipi_{k}', self.tau_r + self.tau_u, f'D_Ip_{k}'))
            coupling.append((f'Ipi_{k}:Ipi_{k}=>Ipd_{k}', 1,f'Ipi_{k}'))
            
            # from Iau to Iad, random + universal testing using intermetiate compartment to allow contact tracing
            coupling.append((f'Iau_{k}:Iau_{k}=>Iai_{k}', self.tau_r + self.tau_u, f'D_Ia_{k}'))         
            coupling.append((f'Iai_{k}:Iai_{k}=>Iad_{k}', 1,f'Iai_{k}'))

            # from Isu to Isd, random + universal + symptomatic testing using intermetiate compartment to allow contact tracing
            coupling.append((f'Isu_{k}:Isu_{k}=>Isi_{k}', self.tau_s + self.tau_r + self.tau_u, f'D_Is_{k}'))
            coupling.append((f'Isi_{k}:Isi_{k}=>Isd_{k}', 1,f'Isi_{k}'))

        return tuple(coupling)

    def run(self, t0, tmax, tsteps, initial_states):
        """
        Run the model from t0 to tmax in tsteps steps, given the
        starting model state.
        """
        y0 = initial_states
        #N = sum(y0)
        #y0 = y0/N
        self.cm = modCModel(n_age=len(self.age_groups), states=self.states)

        
        for desc, rate, name in self.couplings():
            self.cm.set_coupling_rate(desc, rate, name=name)

        t = np.linspace(t0, tmax, tsteps+1)
        cl = list(self.cm.couplings)
        index_matrix_cou = np.array([[cl.index(c + k) for c in ['D_Ip_','D_Ia_','D_Is_']] for k in self.age_groups])
        index_matrix_com = np.array([[self.states.index(c + k) for c in ['Ipi_','Iai_','Isi_']] for k in self.age_groups])
        traj = self.cm.integrate(t, y0, index_matrix_cou, index_matrix_com, self.tsp*self.Cp**(-1), self.tsp*self.Ca**(-1), self.tsp*self.Cs**(-1), ivpargs={"max_step": 1.0})

        return (t, traj["y"])

def runModel(model, t0, tmax, steps, parameters={}, initial={}, seed=0, **unused):

    """
    Run the provided model with the given parameters, initial conditions and
    interventions. The model is run until tmax
      - `model` is a model class (not an instance). It will be initialised by
         this function.
      - `t0` the start time of the simulation. This will usually be 0.
      - `tmax` the end time of the simulation.
      - `steps` the number of time-steps to report, evenly spaced from `t0`
         to `tmax`.
      - `parameters` model parameters. This is a dictionary of the form,
           { "beta": 0.033, "c": 13, "theta": 0.1 }
      - `initial` initial conditions. For example,
           { "N": 10000, "IU": 10, "EU": 5 }
      - `seed` the random seed to set at the beginning of the simulation.
    Returns a tuple `(t, traj)` where `t` is the sequence of
    times, `traj` is the sequence of observables produced by the model
    """

    np.random.seed(seed)

    m = model()
    m.set_parameters(**dict((k, parameters[k]["value"])
                                   for k in parameters.keys()))
    state = m.initial_conditions(**dict((k, initial[k]["value"])
                                   for k in initial.keys()))

    log.info("Running model: {}".format(m.name))
    log.info("Random seed: {}".format(seed))
    log.info("Parameters: {}".format(parameters))
    log.info("Initial conditions: {}".format(initial))
    
    t, traj = m.run(t0, tmax, steps, state)

    return t, traj, m.states
