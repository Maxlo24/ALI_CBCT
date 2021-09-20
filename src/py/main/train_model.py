from model import *
from train_utils import *

import argparse

import glob
from sklearn.model_selection import train_test_split
import logging
import sys



def main(args):

    # #####################################
    #  Init_param
    # #####################################
    label_nbr = 3
    nbr_workers = 4 

    spacing = args.spacing
    cropSize = args.crop_size

    scan_lst = []
    label_lst = []

    datalist = []

    # #####################################
    #  Get data
    # #####################################

    scan_normpath = os.path.normpath("/".join([args.dir_scans, '**', '']))
    for img_fn in sorted(glob.iglob(scan_normpath, recursive=True)):
        #  print(img_fn)
        if os.path.isfile(img_fn) and True in [ext in img_fn for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:
            scan_lst.append(img_fn)

    label_normpath = os.path.normpath("/".join([args.dir_landmarks, '**', '']))
    for img_fn in sorted(glob.iglob(label_normpath, recursive=True)):
        #  print(img_fn)
        if os.path.isfile(img_fn) and True in [ext in img_fn for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:
            label_lst.append(img_fn)
    
    if len(scan_lst) != len(label_lst):
        print("ERROR : Not the same number of scan and landmark file")
        return


    for file_id in range(0,len(scan_lst)):
        data = {"image" : scan_lst[file_id], "label" : label_lst[file_id]}
        datalist.append(data)


    trainingSet, validationSet = train_test_split(datalist, test_size=args.test_percentage/100, random_state=len(datalist))  

    if not os.path.exists(args.dir_model):
        os.makedirs(args.dir_model)

    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    root_dir = tempfile.mkdtemp() if directory is None else directory

    print("WORKING IN : ", root_dir)

    train_transforms = createTrainTransform(spacing,cropSize,args.dir_cash)
    val_transforms = createValidationTransform(spacing,args.dir_cash)

    print(trainingSet)
    print(validationSet)

    # #####################################
    #  Load data
    # #####################################

    print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    train_ds = CacheDataset(
        data=trainingSet,
        transform=train_transforms,
        # cache_num=24,
        cache_rate=1.0,
        num_workers=nbr_workers,
    )
    train_loader = DataLoader(
        train_ds, batch_size=5, shuffle=True, num_workers=nbr_workers, pin_memory=True
    )

    val_ds = CacheDataset(
        data=validationSet,
        transform=val_transforms, 
        # cache_num=6, 
        cache_rate=1.0, 
        num_workers=nbr_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=2, shuffle=False, num_workers=nbr_workers, pin_memory=True
    )

    # #####################################
    #  Training
    # #####################################

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Create_UNETR(
        label_nbr=label_nbr,
        cropSize=cropSize
    ).to(device)

    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    torch.backends.cudnn.benchmark = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    model_data = {
        "model" : model,
        "name": args.model_name,
        "dir":args.dir_model, 
        "loss_f":loss_function, 
        "optimizer":optimizer 
        }

    max_iterations = args.max_iterations
    eval_num = int(args.max_iterations/50)
    post_label = AsDiscrete(to_onehot=True, n_classes=label_nbr)
    post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=label_nbr)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0
    epoch_loss_values = []
    metric_values = []
    while global_step < max_iterations:
        global_step, dice_val_best, global_step_best = train(
            data_model=model_data,
            cropSize=cropSize,
            global_step=global_step,
            eval_num=eval_num,
            max_iterations=max_iterations,
            train_loader=train_loader,
            val_loader=val_loader,
            epoch_loss_values=epoch_loss_values,
            metric_values=metric_values,
            dice_val_best=dice_val_best,
            global_step_best=global_step_best,
            dice_metric=dice_metric,
            post_pred=post_pred,
            post_label=post_label
        )

    print(
    f"train completed, best_metric: {dice_val_best:.4f} "
    f"at iteration: {global_step_best}"
    )
    print("Best model at : ", model_data["best"])

    if directory is None:
        shutil.rmtree(root_dir)



# #####################################
#  Args
# #####################################

if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='MD_reader', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    input_group = parser.add_argument_group('dir')
    input_group.add_argument('--dir_project', type=str, help='Directory with all the project',default='/Users/luciacev-admin/Documents/Projects/ALI_benchmark')
    input_group.add_argument('--dir_data', type=str, help='Input directory with 3D images', default=parser.parse_args().dir_project+'/data')
    input_group.add_argument('--dir_scans', type=str, help='Input directory with the scans',default=parser.parse_args().dir_data+'/Scans')
    input_group.add_argument('--dir_landmarks', type=str, help='Input directory with the landmarks',default=parser.parse_args().dir_data+'/Landmarks')
    input_group.add_argument('--dir_cash', type=str, help='Output directory of the training',default=parser.parse_args().dir_data+'/Cash')
    input_group.add_argument('--dir_model', type=str, help='Output directory of the training',default=parser.parse_args().dir_data+'/ALI_models')



    input_group.add_argument('-sp', '--spacing', nargs="+", type=float, help='Wanted output x spacing', default=[2,2,2])
    input_group.add_argument('-cs', '--crop_size', nargs="+", type=float, help='Wanted crop size', default=[64,64,64])
    input_group.add_argument('-mi', '--max_iterations', type=int, help='Number of training epocs', default=25000)
    input_group.add_argument('-tp', '--test_percentage', type=int, help='Percentage of data to keep for validation', default=20)
    input_group.add_argument('-mn', '--model_name', type=str, help='Name of the model', default="ALI_model")



    args = parser.parse_args()
    
    main(args)