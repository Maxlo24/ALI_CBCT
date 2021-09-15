from train_utils import *
import argparse
import glob



def main(args):

    label_nbr = 3

    scan_lst = []
    label_lst = []

    datalist = []

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
        data = {}
        data["image"],data["label"] = scan_lst[file_id],label_lst[file_id]
        datalist.append(data)

    val_file_nbr = int(len(datalist)*(33/100)) + 1
    # print(val_file_nbr)
    
    trainingSet = datalist[val_file_nbr:]
    validationSet = datalist[0:val_file_nbr]


    if not os.path.exists(args.dir_out):
        os.makedirs(args.dir_out)

    # directory = os.environ.get("MONAI_DATA_DIRECTORY")
    # root_dir = tempfile.mkdtemp() if directory is None else directory
    root_dir = args.dir_out
    print("WORKING IN : ", root_dir)

    spacing = args.spacing
    cropSize = args.crop_size

    train_transforms = createTrainTransform(spacing,cropSize)
    val_transforms = createValidationTransform(spacing)

    print(trainingSet)
    print(validationSet)


    train_ds = CacheDataset(
        data=trainingSet,
        transform=train_transforms,
        cache_num=24,
        cache_rate=1.0,
        num_workers=1,
    )
    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True, num_workers=1, pin_memory=True
    )

    val_ds = CacheDataset(
        data=validationSet, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNETR(
        in_channels=1,
        out_channels=label_nbr,
        img_size=cropSize,
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.1,
    ).to(device)

    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    torch.backends.cudnn.benchmark = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    max_iterations = args.max_iterations
    eval_num = int(args.max_iterations/10)
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
            model=model,
            cropSize=cropSize,
            loss_function=loss_function,
            optimizer=optimizer,
            global_step=global_step,
            eval_num=eval_num,
            max_iterations=max_iterations,
            train_loader=train_loader,
            val_loader=val_loader,
            epoch_loss_values=epoch_loss_values,
            metric_values=metric_values,
            dice_val_best=dice_val_best,
            global_step_best=global_step_best,
            root_dir=root_dir,
            dice_metric=dice_metric,
            post_pred=post_pred,
            post_label=post_label
        )
    model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
    
    print(
    f"train completed, best_metric: {dice_val_best:.4f} "
    f"at iteration: {global_step_best}"
    )

    # ShowTrainResult(
    #     dice_val_best=dice_val_best,
    #     global_step_best=global_step_best,
    #     eval_num=eval_num,
    #     epoch_loss_values=epoch_loss_values,
    #     metric_values=metric_values
    # )


    case_num = 4
    model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
    model.eval()
    with torch.no_grad():
        img_name = os.path.split(val_ds[case_num]["image_meta_dict"]["filename_or_obj"])[1]
        img = val_ds[case_num]["image"]
        label = val_ds[case_num]["label"]
        val_inputs = torch.unsqueeze(img, 1).cuda()
        val_labels = torch.unsqueeze(label, 1).cuda()
        val_outputs = sliding_window_inference(
            val_inputs, cropSize, 4, model, overlap=0.8
        )
        print(val_outputs)
        # plt.figure("check", (18, 6))
        # plt.subplot(1, 3, 1)
        # plt.title("image")
        # plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")
        # plt.subplot(1, 3, 2)
        # plt.title("label")
        # plt.imshow(val_labels.cpu().numpy()[0, 0, :, :, slice_map[img_name]])
        # plt.subplot(1, 3, 3)
        # plt.title("output")
        # plt.imshow(
        #     torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice_map[img_name]]
        # )
        # plt.show()


    # if directory is None:
    #     shutil.rmtree(root_dir)










if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='MD_reader', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    input_group = parser.add_argument_group('dir')
    input_group.add_argument('--dir_project', type=str, help='Directory with all the project',default='/Users/luciacev-admin/Documents/Projects/ALI_benchmark')
    input_group.add_argument('--dir_data', type=str, help='Input directory with 3D images', default=parser.parse_args().dir_project+'/data')
    input_group.add_argument('--dir_scans', type=str, help='Input directory with the scans',default=parser.parse_args().dir_data+'/Scans')
    input_group.add_argument('--dir_landmarks', type=str, help='Input directory with the landmarks',default=parser.parse_args().dir_data+'/Landmarks')
    input_group.add_argument('--dir_out', type=str, help='Output directory of the training',default=parser.parse_args().dir_data+'/Out')


    input_group.add_argument('-sp', '--spacing', nargs="+", type=float, help='Wanted output x spacing', default=[2,2,2])
    input_group.add_argument('-cs', '--crop_size', nargs="+", type=float, help='Wanted crop size', default=[64,64,64])
    input_group.add_argument('-mi', '--max_iterations', type=int, help='number of training epocs', default=250)

    args = parser.parse_args()
    
    main(args)