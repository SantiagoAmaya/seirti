import argparse
import yaml
from seirti_ode import runModel, SEIRTIODE
import matplotlib.pyplot as plt
import numpy as np

def seirti():

    parser = argparse.ArgumentParser("seirti",
                                      description="Population-wide Testing, Tracing and Isolation Models", 
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-y','--yaml', help='The file used for yaml parser', required=True)
    parser.add_argument('-n','--name', help='Name of the output file')
    parser.add_argument('-s','--stochastic', action="store_true", help='If the model will be run stochastcially')
    args = parser.parse_args()

    with open(args.yaml, 'r') as stream:
        yamlf = yaml.load(stream, yaml.FullLoader)

    name = args.name if args.name else 'res.npz'
    np.savez(name, *runModel(SEIRTIODE, args.stochastic, 0,300,301,**yamlf))

if __name__ == '__main__':
    seirti()