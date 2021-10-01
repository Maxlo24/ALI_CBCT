
import argparse
import glob
import sys
import os





def main(args):
	img_fn_array = []
	outpath = os.path.normpath("/".join([args.out]))

	if args.image:
		img_obj = {}
		img_obj["img"] = args.image
		img_obj["out"] = outpath
		img_fn_array.append(img_obj)
		
		
	if args.dir:
		normpath = os.path.normpath("/".join([args.dir, '**', '']))
		for img_fn in glob.iglob(normpath, recursive=True):
			if os.path.isfile(img_fn) and True in [ext in img_fn for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:
				img_obj = {}
				img_obj["img"] = img_fn
				img_obj["out"] = outpath + img_fn.replace(args.dir,'')
				img_fn_array.append(img_obj)
				
	print("Rescale with spacing :", args.spacing)
	for img_obj in img_fn_array:
		image = img_obj["img"]
		out = img_obj["out"]
		
		if not os.path.exists(os.path.dirname(out)):
			os.makedirs(os.path.dirname(out))
		# SetSpacing(image, out, args.spacing)
		

if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='MD_reader', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    input_group = parser.add_argument_group('Input files')
    input_group.add_argument('-i','--dir', type=str, help='Input directory with 3D images',required=True)

    output_params = parser.add_argument_group('Output parameters')
    output_params.add_argument('-o','--out', type=str, help='Output directory', required=True)

    input_group.add_argument('-reg','--select_region',nargs="+",type=str,help="Prepare the data for uper and/or lower landmark training (ex: u l cb)", default=[])

    input_group.add_argument('-bd', '--box_dist', type=int, help='distance between 2 ROI', default=5)
    
    args = parser.parse_args()
    
    main(args)