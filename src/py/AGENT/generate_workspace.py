from utils import*
import argparse
import glob
import sys
import os
import shutil
import random

def main(args):

    dir = args.input_dir
    out_dir = args.out
    test_percentage = args.test_percentage
    
    data_lst = []
    normpath = os.path.normpath("/".join([dir, '**', '']))
    for img_fn in sorted(glob.iglob(normpath, recursive=True)):
        #  print(img_fn)
        basename = os.path.basename(img_fn)


        if True in [ext in basename for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz",".json"]]:
            dir_patient = os.path.dirname(img_fn)

            if dir_patient not in data_lst:
                data_lst.append(dir_patient)

    random.shuffle(data_lst)
    nbr_cv_fold = int(1/test_percentage)
    nbr_cv_fold = 1
    for i in range(nbr_cv_fold):

        cv_dir_out =  os.path.join(out_dir,"CV_fold_" + str(i))
        if not os.path.exists(cv_dir_out):
            os.makedirs(cv_dir_out)

        data_fold =  os.path.join(cv_dir_out,"data")
        if not os.path.exists(data_fold):
            os.makedirs(data_fold)
        
        patients_fold =  os.path.join(data_fold,"Patients")
        if not os.path.exists(patients_fold):
            os.makedirs(patients_fold)
    
        test_fold =  os.path.join(data_fold,"test")
        if not os.path.exists(test_fold):
            os.makedirs(test_fold)

        len_lst = len(data_lst)
        len_test = int(len_lst*test_percentage)
        start = i*len_test
        end = (i+1)*len_test
        if end > len_lst: end = len_lst

        # print(data_lst)
        # print(len_lst)
        # print(start,end)
        training_patients = data_lst[:start] + data_lst[end:]
        test_patients = data_lst[start:end]

        for folder in training_patients:
            shutil.copytree(folder, patients_fold+"/"+ os.path.basename(folder)) 

        for folder in test_patients:
            shutil.copytree(folder, test_fold+"/"+ os.path.basename(folder)) 
        # train_cv_dir_out =  os.path.join(patients_fold,folder)
        # if not os.path.exists(train_cv_dir_out):
        #     os.makedirs(train_cv_dir_out)

        # for patient in training_patients:
        #     shutil.copyfile(patient["scan"], os.path.join(train_cv_dir_out,os.path.basename(patient["scan"])))
        #     shutil.copyfile(patient["seg"], os.path.join(train_cv_dir_out,os.path.basename(patient["seg"])))

        # test_cv_dir_out =  os.path.join(test_fold,folder)
        # if not os.path.exists(test_cv_dir_out):
        #     os.makedirs(test_cv_dir_out)

        # for patient in test_patients:
        #     shutil.copyfile(patient["scan"], os.path.join(test_cv_dir_out,os.path.basename(patient["scan"])))
        #     shutil.copyfile(patient["seg"], os.path.join(test_cv_dir_out,os.path.basename(patient["seg"])))


        # print(training_patients)
        # print(test_patients)

if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='Initialise data to be ready for training the U, L and CB landmarks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    input_group = parser.add_argument_group('Input files')
    input_group.add_argument('-i','--input_dir', type=str, help='Input directory with 3D images',required=True)
    input_group.add_argument('-tp', '--test_percentage', type=float, help='Test porcentage', default=0.2)

    output_params = parser.add_argument_group('Output parameters')
    output_params.add_argument('-o','--out', type=str, help='Output directory', required=True)


    args = parser.parse_args()
    
    main(args)
