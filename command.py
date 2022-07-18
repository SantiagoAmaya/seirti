import argparse
import yaml
from seirti_ode import runModel, SEIRTIODE
import matplotlib.pyplot as plt
import numpy as np

def command():

    parser = argparse.ArgumentParser("seirti",
                                      description="Population-wide Testing, Tracing and Isolation Models", 
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--yaml', help='The file used for yaml parser', required=True)
    args = parser.parse_args()

    with open(args.yaml, 'r') as stream:
        yamlf = yaml.load(stream, yaml.FullLoader)

    
    t, traj, states = runModel(SEIRTIODE,0,100,1000,**yamlf)
    res = {s:traj[:,i] for i,s in enumerate(states)}
    # states to plot in ax0 and ax1 respectively
    st1 = ['S_0_18','Eu_0_18','Ipu_0_18','Iau_0_18','Isu_0_18','R_0_18']
    st2 = ['S_0_18','Ed_0_18','Ipd_0_18','Iad_0_18','Isd_0_18','R_0_18']

    fig, ax = plt.subplots(1,2,figsize= (20,10))

    ax[0].set_xlabel('t')
    ax[0].set_ylabel('Population (%)')
    ax[0].set_ylim(0, 35)

    ax[1].set_xlabel('t')
    ax[1].set_ylabel('Population (%)')
    ax[1].set_ylim(0, 35)

    for n,m in zip(st1,st2):
        ax[0].plot(t, res[n]*100, label=n)
        ax[1].plot(t, res[m]*100, label=m)
        
    ax[0].legend()
    ax[1].legend()
    plt.savefig('foo.png', bbox_inches='tight')



if __name__ == '__main__':
    command()