from utils import (
    GetEnvironmentLst,
)


from GlobalVar import*

from Environement_class import (Environement)
import os

import argparse
from multiprocessing import Pool
def main(args):

    # #####################################
    #  Init_param
    # #####################################

    environments_param = {
        "type" : Environement,
        "dir" : args.dir_input,
        "spacings" : args.spacing,
        "padding" : np.array([1,1,1]),
        "landmarks" : args.landmarks,
        "verbose" : False,
        "rotated" : False
    }

    env_lst = GetEnvironmentLst(environments_param)

    id = "rot0"
    for env in env_lst:
        out_dir = os.path.dirname(env.images_path[0])
        out_dir = out_dir.replace(args.dir_input,args.dir_output)
        out_dir = os.path.join(out_dir,id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        env.SaveEnvironmentState(out_dir,id)
        # print(env.dim_landmarks)

    for rotation in range(args.nbr_rotataion):
        id = "rot" + str(rotation+1)
        for env in env_lst:
            env.SetRandomRotation()
            out_dir = os.path.dirname(env.images_path[0])
            out_dir = out_dir.replace(args.dir_input,args.dir_output)
            out_dir = os.path.join(out_dir,id)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            env.SaveEnvironmentState(out_dir,id)

# #####################################
#  Args
# #####################################

if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='Training for Automatic Landmarks Identification', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    input_group = parser.add_argument_group('dir')
    input_group.add_argument('-i','--dir_input', type=str, help='Input directory with the scans',default='/Users/luciacev-admin/Documents/Projects/Benchmarks/MSDRL_benchmark/data/patients')
    input_group.add_argument('-o','--dir_output', type=str, help='Input directory with the scans',default='/Users/luciacev-admin/Desktop/test')

    #Environment
    input_group.add_argument('-lm','--landmarks',nargs="+",type=str,help="Prepare the data for uper and/or lower landmark training (ex: U L CB)", default=["U","L","CB"])
    input_group.add_argument('-sp', '--spacing', nargs="+", type=float, help='Spacing of the different scales', default=[1,0.3])
    
    #Training data
    input_group.add_argument('-nr', '--nbr_rotataion', type=int, help='Rotataion number', default=10)

    args = parser.parse_args()
    
    main(args)