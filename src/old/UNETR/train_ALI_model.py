from matplotlib.pyplot import step
from numpy import imag
from model import *
from utils import *

import argparse

import logging
import sys

def main(args):

    # #####################################
    #  Init_param
    # #####################################
    label_nbr = args.nbr_label
    nbr_workers = args.nbr_worker

    spacing = args.spacing
    cropSize = args.crop_size

    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    root_dir = tempfile.mkdtemp() if directory is None else directory

    print("WORKING IN : ", root_dir)

    print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    datalist = GetDataList(
        dirDict = {
            "image" : args.dir_scans,
            "landmarks" : args.dir_landmarks,
            "ROI" : args.dir_ROI
        }
    )

    # label_nbr = GetSegLabelNbr(datalist[0]["landmarks"])

    trainingSet, validationSet = train_test_split(datalist, test_size=args.test_percentage/100, random_state=len(datalist))  

    if not os.path.exists(args.dir_model):
        os.makedirs(args.dir_model)


    train_transforms = CreateALITrainTransform()
    val_transforms = CreateALIValidationTransform()

    # print(len(trainingSet))
    # print(trainingSet)
    # print(len(validationSet))
    # print(validationSet)

    # #####################################
    #  Load data
    # #####################################

    replace_rate = 1.0
    train_batch_size = 10
    val_batch_size = 5

    train_ds = CacheDataset(
        data=trainingSet,
        transform=train_transforms,
        # cache_num=24,
        cache_rate=1.0,
        num_workers=nbr_workers,
        # replace_rate = replace_rate,
        # cache_num = train_cache_num
    )

    # x = train_transforms(trainingSet[0])

    train_loader = DataLoader(
        train_ds, batch_size=train_batch_size, shuffle=True, num_workers=nbr_workers, pin_memory=True
    )

    val_ds = CacheDataset(
        data=validationSet,
        transform=val_transforms, 
        # cache_num=6, 
        cache_rate=1.0, 
        num_workers=nbr_workers,
        # replace_rate = replace_rate,
        # cache_num = val_cache_num
    )
    val_loader = DataLoader(
        val_ds, batch_size=val_batch_size, shuffle=False, num_workers=nbr_workers, pin_memory=True
    )


    # for data in datalist:
    #     transformed_data = train_transforms(data)
    #     print(data["image"])
    #     print("image",transformed_data["image"].size())
    #     print("ROI  ",transformed_data["ROI"].size())
    #     print("LM   ",transformed_data["landmarks"].size())

    # #####################################
    #  Training
    # #####################################

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Create_UNETR(
        input_channel=2,
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
    step_to_val = 0
    epoch_loss_values = []
    metric_values = []
    while global_step < max_iterations:
        if (step_to_val >= eval_num) or global_step >= max_iterations:
            dice_val_best, global_step_best = validate(
                inID=["image","ROI"],
                outID = "landmarks",
                data_model=model_data,
                val_loader = val_loader,
                cropSize=cropSize,
                global_step=global_step,
                metric_values=metric_values,
                dice_val_best=dice_val_best,
                global_step_best=global_step_best,
                dice_metric=dice_metric,
                post_pred=post_pred,
                post_label=post_label
            )
            step_to_val -= eval_num

        steps = train(
            inID=["image","ROI"],
            outID = "landmarks",
            data_model=model_data,
            global_step=global_step,
            epoch_loss_values=epoch_loss_values,
            max_iterations=max_iterations,
            train_loader=train_loader,
        )
        global_step += steps
        step_to_val += steps

    dice_val_best, global_step_best = validate(
        inID=["image","ROI"],
        outID = "landmarks",
        data_model=model_data,
        val_loader = val_loader,
        cropSize=cropSize,
        global_step=global_step,
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

    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    if directory is None:
        shutil.rmtree(root_dir)

# #####################################
#  Args
# #####################################

if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='Training for Automatic Landmarks Identification', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    input_group = parser.add_argument_group('dir')
    input_group.add_argument('--dir_project', type=str, help='Directory with all the project',default='/Users/luciacev-admin/Documents/Projects/ALI_benchmark')
    input_group.add_argument('--dir_data', type=str, help='Input directory with 3D images', default=parser.parse_args().dir_project+'/data')
    input_group.add_argument('--dir_scans', type=str, help='Input directory with the scans',default=parser.parse_args().dir_data+'/Crop/Scans')
    input_group.add_argument('--dir_landmarks', type=str, help='Input directory with the landmarks',default=parser.parse_args().dir_data+'/Crop/Segs')
    input_group.add_argument('--dir_ROI', type=str, help='Input directory with the ROI',default=parser.parse_args().dir_data+'/Crop//ROI')

    input_group.add_argument('--dir_cash', type=str, help='Output directory of the training',default=parser.parse_args().dir_data+'/Cash')
    input_group.add_argument('--dir_model', type=str, help='Output directory of the training',default=parser.parse_args().dir_data+'/ALI_models')

    input_group.add_argument('-sp', '--spacing', nargs="+", type=float, help='Wanted output x spacing', default=[0.5,0.5,0.5])
    input_group.add_argument('-cs', '--crop_size', nargs="+", type=float, help='Wanted crop size', default=[64,64,64])
    input_group.add_argument('-mi', '--max_iterations', type=int, help='Number of training epocs', default=25000)
    input_group.add_argument('-tp', '--test_percentage', type=int, help='Percentage of data to keep for validation', default=20)
    input_group.add_argument('-mn', '--model_name', type=str, help='Name of the model', default="ALI_model")
    input_group.add_argument('-nl', '--nbr_label', type=int, help='Number of label', default=4)
    input_group.add_argument('-nw', '--nbr_worker', type=int, help='Number of worker', default=0)

    args = parser.parse_args()
    
    main(args)