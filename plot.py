import matplotlib.pyplot as plt
import numpy as np
import argparse



def plot():

    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--name', type=str, help='name of the output image file')
    parser.add_argument('-i','--input', required = True, type=str, help='name of the output image file')
    parser.add_argument('-s','--stochastic', action="store_true", help='If the model is stochastic')
    parser.add_argument('-o','--observables', required=True, nargs='+', type=str, help='what states to plot e.g. S Eu Ed')
    parser.add_argument('-a','--age', required=True, type = str, help='age groups of the states to plot in format e.g. _0_18')
    args = parser.parse_args()

    container = np.load(args.input)
    t,traj,states = [container[key] for key in container]

    res = {s:traj[...,i] for i,s in enumerate(states)}
    # states to plot
    
    st1 = [o + args.age for o in args.observables]

    if args.stochastic:

        fig, ax = plt.subplots(1,1,figsize= (10,10))

        for i, n in enumerate(st1):

            for r in res[n]:
                ax.plot(t, r[:-1], c=f'C{i}', alpha=0.1)
                
            ax.plot(t, r[:-1], c=f'C{i}', alpha=0.1, label=n)       

          
        ax.set_xlabel('t')
        ax.set_ylabel('Population')
        #ax.set_ylim(0, 100)        
            
        ax.legend()
        if args.name:
            name = args.name
        else:
            name = 'bin_chain.png'
        plt.savefig(name, bbox_inches='tight')

    else:

        fig, ax = plt.subplots(1,1,figsize= (10,10))

        ax.set_xlabel('t')
        ax.set_ylabel('Population ')
        #ax.set_ylim(0, 100)

        for n in st1:
            ax.plot(t, res[n], label=n)
            
        ax.legend()
        if args.name:
            name = args.name
        else:
            name = 'ode.png'
        plt.savefig(name, bbox_inches='tight')


if __name__ == '__main__':
    plot()