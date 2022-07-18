__all__ = ['SEIRTIODE']

from typing_extensions import Self
import numpy as np
from cpyment import CModel
from model import Model
import yaml
import logging

log = logging.getLogger(__name__)

class SEIRTIODE:
    name = "SEIR-TI ODE"
    # esto lo puedo incluir en el yaml
    states_per_age = ['Su','Sd','Eu','Ed','Ipu','Ipd','Iau','Iad','Isu','Isd','Ru','Rd']    

    def set_parameters(self, **params):
        """
        Set the model parameters
        """
        for k, v in params.items():
            setattr(self, k, v)
        self.Ca = np.array(self.Ca)
        self.Cs = np.array(self.Cs)
        self.states = [s + '_' + k for k in self.age_groups for s in self.states_per_age]

    def initial_conditions(self, **initial):
                
        y0 = np.zeros(len(self.states))
        for k, v in initial.items():
            y0[self.states.index(k)] = v
        return y0


    def couplings(self):
       # Now add the coupling
        coupling = []
        for i,k in enumerate(self.age_groups):
            
            # from Sd to Su
            coupling.append((f'Sd_{k}:Sd_{k}=>Su_{k}', self.kappa, f'kappa_s_{k}'))
            
            # from Su to Sd, strategic testing
            coupling.append((f'Su_{k}:Su_{k}=>Sd_{k}', self.tau_s, f'tau_s_{k}'))
            
            # from Eu to Ed, strategic testing
            coupling.append((f'Eu_{k}:Eu_{k}=>Ed_{k}', self.tau_s, f'tau_E_{k}'))
            
            # from Eu to Ipu
            coupling.append((f'Eu_{k}:Eu_{k}=>Ipu_{k}', self.gamma, f'gamma_u_{k}'))
            
            # from Ed to Ipd
            coupling.append((f'Ed_{k}:Ed_{k}=>Ipd_{k}', self.gamma, f'gamma_d_{k}'))
            
            # from Ipu to Iau
            coupling.append((f'Ipu_{k}:Ipu_{k}=>Iau_{k}', self.p*self.theta, f'p_u_{k}'))
            
            # from Ipu to Isu
            coupling.append((f'Ipu_{k}:Ipu_{k}=>Isu_{k}', (1-self.p)*self.theta, f'q_u_{k}'))
            
            # from Ipu to Ipd, strategic testing
            coupling.append((f'Ipu_{k}:Ipu_{k}=>Ipd_{k}', self.tau_s, f'tau_ip_{k}'))

            # from Ipd to Iad
            coupling.append((f'Ipd_{k}:Ipd_{k}=>Iad_{k}', self.p*self.theta, f'p_d_{k}'))
            
            # from Ipd to Isd
            coupling.append((f'Ipd_{k}:Ipd_{k}=>Isd_{k}', (1-self.p)*self.theta, f'q_d_{k}'))
            
            # from Iau to Iad, strategic testing
            coupling.append((f'Iau_{k}:Iau_{k}=>Iad_{k}', self.tau_s, f'tau_ia_{k}'))
            
            # from Iau to Ru
            coupling.append((f'Iau_{k}:Iau_{k}=>Ru_{k}', self.delta1, f'delta1_u_{k}'))
            
            # from Iad to Rd
            coupling.append((f'Iad_{k}:Iad_{k}=>Rd_{k}', self.delta1, f'delta1_d_{k}'))
            
            # from Isu to Ru
            coupling.append((f'Isu_{k}:Isu_{k}=>Ru_{k}', self.delta2, f'delta2_u_{k}'))
            
            # from Isd to Rd
            coupling.append((f'Isd_{k}:Isd_{k}=>Rd_{k}', self.delta2, f'delta2_d_{k}'))
            
            # from Isu to Isd, strategic testing
            coupling.append((f'Isu_{k}:Isu_{k}=>Isd_{k}', self.tau_s, f'tau_is_{k}'))
            
            # from Ru to Rd, strategic testing
            coupling.append((f'Ru_{k}:Ru_{k}=>Rd_{k}', self.tau_s, f'tau_r_{k}'))
            
            # from Rd to Ru
            coupling.append((f'Rd_{k}:Rd_{k}=>Ru_{k}', self.kappa, f'kappa_r_{k}'))      
    
            for j,k_ in enumerate(self.age_groups):            
            
                # from Su to Eu
                coupling.append((f'Su_{k}*Ipu_{k_}:Su_{k}=>Eu_{k}', self.q*self.Ca[i,j]**(-1), f'beta_pu({i},{j})'))
                coupling.append((f'Su_{k}*Iau_{k_}:Su_{k}=>Eu_{k}', self.q*self.Ca[i,j]**(-1), f'beta_au({i},{j})'))
                coupling.append((f'Su_{k}*Isu_{k_}:Su_{k}=>Eu_{k}', self.q*self.Cs[i,j]**(-1), f'beta_su({i},{j})'))
                # from Sd to Eu
                coupling.append((f'Sd_{k}*Ipu_{k_}:Sd_{k}=>Eu_{k}', self.q*self.Ca[i,j]**(-1), f'beta_pd({i},{j})'))
                coupling.append((f'Sd_{k}*Iau_{k_}:Sd_{k}=>Eu_{k}', self.q*self.Ca[i,j]**(-1), f'beta_ad({i},{j})'))
                coupling.append((f'Sd_{k}*Isu_{k_}:Sd_{k}=>Eu_{k}', self.q*self.Cs[i,j]**(-1), f'beta_sd({i},{j})'))

        return tuple(coupling)

    def reset_parameters(self, **params):
        self.set_parameters(**params)
        for _,rate, name in self.couplings(self.age_groups):
            self.cm.edit_coupling_rate(name, rate)

    def run(self, t0, tmax, tsteps, initial_states):
        """
        Run the model from t0 to tmax in tsteps steps, given the
        starting model state.
        """
        y0 = initial_states
        N = sum(y0)
        #y0 = y0/N
        self.cm = CModel(self.states)
        for desc, rate, name in self.couplings():
            self.cm.set_coupling_rate(desc, rate, name=name)

        t = np.linspace(t0, tmax, tsteps+1)

        traj = self.cm.integrate(t, y0, ivpargs={"max_step": 1.0})

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
    m.set_parameters(**dict((k, parameters[k]["default"])
                                   for k in parameters.keys()))
    state = m.initial_conditions(**dict((k, initial[k]["value"])
                                   for k in initial.keys()))

    log.info("Running model: {}".format(m.name))
    log.info("Random seed: {}".format(seed))
    log.info("Parameters: {}".format(parameters))
    log.info("Initial conditions: {}".format(initial))

    # piece-wise simulation segments
    times = []
    trajs = []
    
    t, traj = m.run(t0, tmax, steps, state)

    return t, traj, m.states
