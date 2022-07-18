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

    fig, ax = plt.subplots(1,2,figsize= (20,10))

    ax[0].set_xlabel('t')
    ax[0].set_ylabel('Population (%)')
    ax[0].set_ylim(0, 35)

    ax[1].set_xlabel('t')
    ax[1].set_ylabel('Population (%)')
    ax[1].set_ylim(0, 35)

    for i,l in zip(np.arange(0,12,2),states[0:12:2]):
        ax[0].plot(t, traj[:,i]*100, label=l)
        
    for i,l in zip(np.arange(1,13,2),states[1:13:2]):
        ax[1].plot(t, traj[:,i]*100, label=l)
        
    ax[0].legend()
    ax[1].legend()
    plt.savefig('foo.png', bbox_inches='tight')



if __name__ == '__main__':
    command()