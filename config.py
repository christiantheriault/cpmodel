import argparse

# Function to define and load all default parameters
def parse_args():    
        
    # LOAD DEFAULT PARAMETERS
    N=5
    netruns=1
    
    parser = argparse.ArgumentParser(description='cp model')
    parser.add_argument('--N', type=int, default=N,help='Number of input dimensions for vector stimuli')
    parser.add_argument('--netruns', type=int, default=netruns,help='Number of net runs to average')
    
    return parser.parse_args()


