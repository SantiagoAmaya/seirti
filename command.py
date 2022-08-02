import argparse
import yaml
from seirti_ode import runModel, SEIRTIODE
import matplotlib.pyplot as plt
import numpy as np

def command():

    parser = argparse.ArgumentParser("seirti",
                                      description="Population-wide Testing, Tracing and Isolation Models", 
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-y','--yaml', help='The file used for yaml parser', required=True)
    parser.add_argument('-s','--stochastic', action="store_true", help='If the model will be run stochastcially')
    args = parser.parse_args()

    with open(args.yaml, 'r') as stream:
        yamlf = yaml.load(stream, yaml.FullLoader)

    
    t, traj, states = runModel(SEIRTIODE, args.stochastic, 0,300,301,**yamlf)
    res = {s:traj[...,i] for i,s in enumerate(states)}
    # states to plot in ax0 and ax1 respectively
    st1 = ['S_0_18','Ipu_0_18','Iau_0_18','Isu_0_18','R_0_18']
    st2 = ['S_0_18','Ipd_0_18','Iad_0_18','Isd_0_18','R_0_18']

    if args.stochastic:

        fig, ax = plt.subplots(1,2,figsize= (20,10))

        for i, (n,m) in enumerate(zip(st1,st2)):

            for r in res[n]:
                        ax[0].plot(t, r, c=f'C{i}', alpha=0.1)
            for r in res[m]:
                        ax[1].plot(t, r, c=f'C{i}', alpha=0.1)

        ax[0].plot(t, r, c=f'C{i}', alpha=0.1, label=n)     
        ax[0].set_xlabel('t')
        ax[0].set_ylabel('Population')
        #ax[0].set_ylim(0, 100)

        
        ax[1].plot(t, r, c=f'C{i}', alpha=0.1, label=m)   
        ax[1].set_xlabel('t')
        ax[1].set_ylabel('Population')
        #ax[1].set_ylim(0, 100)
        
            
        ax[0].legend()
        ax[1].legend()
        plt.savefig('bin_chain.png', bbox_inches='tight')

    else:

        fig, ax = plt.subplots(1,2,figsize= (20,10))

        ax[0].set_xlabel('t')
        ax[0].set_ylabel('Population (%)')
        #ax[0].set_ylim(0, 100)

        ax[1].set_xlabel('t')
        ax[1].set_ylabel('Population (%)')
        #ax[1].set_ylim(0, 100)

        for n,m in zip(st1,st2):
            ax[0].plot(t, res[n], label=n)
            ax[1].plot(t, res[m], label=m)
            
        ax[0].legend()
        ax[1].legend()
        plt.savefig('ode.png', bbox_inches='tight')



if __name__ == '__main__':
    command()