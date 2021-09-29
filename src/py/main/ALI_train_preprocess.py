from utils import*

import argparse



def main(args):

    # #####################################
    #  Init_param
    # #####################################

    cropSize = args.crop_size

    datalist = GetDataList(
        dirDict = {
            "image" : args.dir_scans,
            "landmarks" : args.dir_landmarks,
            "label" : args.dir_ROI
        }
    )

    # print(datalist)

    # if not os.path.exists(args.dir_croped_scan):
    #     os.makedirs(args.dir_croped_scan)

    # if not os.path.exists(args.dir_croped_landmarks):
    #     os.makedirs(args.dir_croped_landmarks)

    for data in datalist:
        CropImageFromROIfile(data["image"],data["landmarks"],data["label"],args.dir_croped,cropSize,args.radius)



if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='Training for Automatic Landmarks Identification', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    dir_group = parser.add_argument_group('dir')
    dir_group.add_argument('--dir_project', type=str, help='Directory with all the project',default='/Users/luciacev-admin/Documents/Projects/ALI_benchmark')
    dir_group.add_argument('--dir_data', type=str, help='Input directory with 3D images', default=parser.parse_args().dir_project+'/data')
    dir_group.add_argument('--dir_scans', type=str, help='Input directory with the scans',default=parser.parse_args().dir_data+'/Scans')
    dir_group.add_argument('--dir_landmarks', type=str, help='Input directory with the landmarks',default=parser.parse_args().dir_data+'/Landmarks')
    dir_group.add_argument('--dir_ROI', type=str, help='Input directory with the ROI',default=parser.parse_args().dir_data+'/ROI')

    dir_group.add_argument('--dir_croped', type=str, help='output directory with the croped images',default=parser.parse_args().dir_data+'/Crop')
    # dir_group.add_argument('--dir_croped_scan', type=str, help='output directory with the croped images',default=parser.parse_args().dir_croped+'/Scans')
    # dir_group.add_argument('--dir_croped_landmarks', type=str, help='output directory with the croped images',default=parser.parse_args().dir_croped+'/Landmarks')
    
    options_group = parser.add_argument_group('options')

    options_group.add_argument('-cs', '--crop_size', nargs="+", type=int, help='Wanted crop size', default=[64,64,64])
    options_group.add_argument('-rad', '--radius', type=int, help='minimum space between 2 ROI center', default=4)

    args = parser.parse_args()
    main(args)
