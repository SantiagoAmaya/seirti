__all__ = ['Model', 'Unimplemented']

import logging
from math import floor
from cpyment import CModel
import numpy as np
from collections import OrderedDict, namedtuple
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import minimize
import scipy.stats as stats
from tqdm import tqdm
from numba import jit

log = logging.getLogger(__name__)

class Unimplemented(Exception):
    """
    Exception raised when a subclass of model fails to implement
    a required method.
    """


class modCModel(CModel):
    """
    This class modifies the CModel class such that
    it is possible to implement contact tracing
    """
    def __init__(self, n_age, states=''):
        super().__init__(states)
        self.n_age = n_age # number of age groups
        self.tau_c = np.zeros(self.n_age) # we asume initial value of tracing rate is 0 for each age group

    def dy_dt(self, y, index_matrix_cou, index_matrix_com, Zp, Za, Zs, N):
        """Time derivative from a given state
        Compute the time derivative of the model for a given state vector.
        Arguments:
            y {np.ndarray} -- State vector
            index_matrix_cou {np.ndarray} -- Coupling indices where tau_c will be summed:
                                        [Ipu_0_18=>Ipd_0_18      Iau_0_18=>Iad_0_18         Isu_0_18=>Isd_0_18]
                                        [Ipu_18_40=>Ipd_18_40    Iau_18_40=>Iad_18_40       Isu_18_40=>Isd_18_40]
                                        [Ipu_etc=>Ipd_etc        Isu_etc=>Isd_etc           Iau_etc=>Iad_etc]
            index_matrix_com {np.ndarray} -- intermediate Compartment indices:
                                        [Ipi_0_18     Iai_0_18      Isi_0_18]
                                        [Ipi_18_40    Iai_18_40     Isi_18_40]
                                        [Ipi_etc      Isi_etc       Iai_etc]
            Z_* {matrix} -- Zeta pre/a/symptomatic (tsp*Ca/Cs/Cp). For now we assume there are different contact matrices for a,p and s
        Returns:
            [np.ndarray] -- Time derivative of y
        """
        
        yext = np.concatenate([y, [1]])
        dydt = yext*0.

        C = self._cdata['C']

        for i in range(len(Zp)):
            np.add.at(C, index_matrix_cou[i,:], self.tau_c[i])

        i1, i2, i3, i4 = self._cdata['i'].T 

        cy = C*yext[i1]*yext[i2]
        np.add.at(dydt, i3, -cy)
        np.add.at(dydt, i4,  cy) 

        # Update tau_c
        #self.tau_c = np.matmul(Zp, dydt[index_matrix_com[:,0]]/N) + np.matmul(Za,dydt[index_matrix_com[:,1]]/N) + np.matmul(Zs,dydt[index_matrix_com[:,2]]/N)

        return dydt[:-1]

    def integrate(self, t, y0, index_matrix_cou, index_matrix_com, Zp, Za, Zs, N, events=None, ivpargs={}):
        
        def ode(t, y, imcou, imcom, zp, za, zs, N):
            return self.dy_dt(y, imcou, imcom, zp, za, zs, N)

        sol = solve_ivp(ode, [t[0], t[-1]], y0, t_eval=t,
                        events=events, args=(index_matrix_cou, index_matrix_com, Zp, Za, Zs, N),**ivpargs)
        traj = sol.y.T

        ans = OrderedDict({'y': traj})

        ans['t'] = sol.t
        if events is not None:
            ans['t_events'] = sol.t_events

        return ans

    def binomial_updates(self, y, index_matrix_cou, index_matrix_com, Zp, Za, Zs,N,h):
        yext = np.concatenate([y, [1]])
        dydt = yext*0.

        C = self._cdata['C']
        for i in range(len(Zp)):
            np.add.at(C, index_matrix_cou[i,:], self.tau_c[i])

        i1, i2, i3, i4 = self._cdata['i'].T     
        cy = stats.binom.rvs(np.int32(np.maximum(yext[i1],0)), 1-np.exp(-h*C*np.maximum(yext[i2],0)))  
        np.add.at(dydt, i3, -cy)
        np.add.at(dydt, i4,  cy) 

        # Update tau_c
        # self.tau_c = np.matmul(Zp, dydt[index_matrix_com[:,0]]/N) + np.matmul(Za,dydt[index_matrix_com[:,1]]/N) + np.matmul(Zs,dydt[index_matrix_com[:,2]]/N)

        return dydt[:-1]


    def binomial_chain(self, samples, steps, y0, index_matrix_cou, index_matrix_com, Zp, Za, Zs,N,h):
        bin_traj = []
        for i in tqdm(range(samples)):
            traj = [y0.copy()]
            for j in range(0,steps,1):
                traj.append(traj[j] + self.binomial_updates(traj[j], index_matrix_cou, index_matrix_com, Zp, Za, Zs,N,h))
                
            bin_traj.append(traj)

        return np.array(bin_traj)

            

        