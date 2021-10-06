from model import*
from utils import*

import argparse

def main(args):

    label_nbr = args.nbr_label
    nbr_workers = args.nbr_worker
    spacing = args.spacing
    cropSize = args.crop_size

    scan_lst = []
    datalist = []

    datalist = GetDataList(
        dirDict = {
            "image" : args.dir
        }
    )

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = Create_UNETR(
        input_channel = 1,
        label_nbr=label_nbr,
        cropSize=cropSize
    ).to(device)

    # print("Loading model", args.load_model)
    # net.load_state_dict(torch.load(args.load_model,map_location=device))
    # net.eval()
    # net.double()

    # define pre transforms
    # pre_transforms = createTestTransform(wanted_spacing= args.spacing,outdir=args.out)


    print("Loading data from", args.dir)

    with torch.no_grad():
        for data in datalist:

            pred_img,input_img = CreatePredictTransform(data["image"])
            print(pred_img.size())
            val_inputs = torch.unsqueeze(pred_img, 1)
            print(val_inputs.size())
            val_outputs = val_inputs
            val_outputs = sliding_window_inference(
                inputs= val_inputs,
                roi_size = cropSize, 
                sw_batch_size= nbr_workers, 
                predictor= net, 
                overlap=0.8
            )

            out_img = torch.argmax(val_outputs, dim=1).detach().cpu()
            out_img = out_img.type(torch.int16)
            # print(out_img,np.shape(out_img))

            baseName = os.path.basename(data["image"])
            scan_name= baseName.split(".")
            pred_name = ""
            for i,element in enumerate(scan_name):
                if i == 0:
                    pred_name += element.replace("scan","Pred")
                else:
                    pred_name += "." + element

            input_dir = os.path.dirname(data["image"])
            
            SavePrediction(out_img ,input_img,os.path.join(input_dir,pred_name))
            

    print("Done : " + str(len(datalist)) + " scan segmented")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Landmarks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    input_group = parser.add_argument_group('directory')
    input_group.add_argument('--dir', type=str, help='Input directory with the scans',default=None, required=True)
    input_group.add_argument('--load_model', type=str, help='Path of the model', default=None, required=True)
    # input_group.add_argument('--out', type=str, help='Output directory with the landmarks',default=None)

    
    input_group.add_argument('-sp', '--spacing', nargs="+", type=float, help='Wanted output x spacing', default=[0.5,0.5,0.5])
    input_group.add_argument('-cs', '--crop_size', nargs="+", type=float, help='Wanted crop size', default=[64,64,64])
    input_group.add_argument('-nl', '--nbr_label', type=int, help='Number of label', default=5)
    input_group.add_argument('-nw', '--nbr_worker', type=int, help='Number of worker', default=1)

    args = parser.parse_args()
    
    main(args)
