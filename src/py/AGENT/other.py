import os
import SimpleITK as sitk
import numpy as np
import glob
import json

# #read all nii.gz files in a folder
# def get_files(path):
#     files = []
#     for r, d, f in os.walk(path):
#         for file in f:
#             if '.nii.gz' in file:
#                 files.append(os.path.join(r, file))
#     return files

# list = get_files("/Users/luciacev-admin/Desktop/Canine_training")
# # print(list)

# for image in list:
#     input_img = sitk.ReadImage(image) 
#     img = sitk.GetArrayFromImage(input_img)


#     img_min = np.min(img)
#     img_max = np.max(img)
#     img_range = img_max - img_min
#     if img_range < 10:
#         print("problem on image", image)


lm_to_delete = ["UR6apex", "UR3apex", "U1apex", "UL3apex", "UL6apex","LR6apex","L1apex","LL6apex","UR1A","UR2A","UR3A","UL1A","UL2A","UL3A","UR6_UL6","UR1_UL1"]
lm_to_rename = {
    "UR6d":"UR6DB",
    "UR6m":"UR6MB",
    "UR3tip":"UR3O",
    "UItip":"UR1O",
    "UL3tip":"UL3O",
    "UL6m":"UL6MB",
    "UL6d":"UL6DB",

    "LR6d":"LR6DB",
    "LR6m":"LR6MB",
    "LItip":"LR1O",
    "LL6m":"LL6MB",
    "LL6d":"LL6DB",

    "UR6":"UR6MP",
    "UL6":"UL6MP",
    "UR1":"UR1O",
    "UL1":"UL1O",
    "UR2":"UR2O",
    "UL2":"UL2O",
    "UR3":"UR3IP",
    "UL3":"UL3IP",
}

input_dir = "/Users/luciacev-admin/Desktop/ULCB_dataset"

print("Reading folder : ", input_dir)

fiducial_files = []

normpath = os.path.normpath("/".join([input_dir, '**', '']))
for img_fn in sorted(glob.iglob(normpath, recursive=True)):
    #  print(img_fn)
    if ".mrk.json" in img_fn:
        fiducial_files.append(img_fn)


for fiducial_file in fiducial_files:
    with open(fiducial_file) as f:
        data = json.load(f)

    control_points = data["markups"][0]["controlPoints"]
    new_control_Points = []
    for control_point in control_points:
        if control_point["label"] not in lm_to_delete:
            if control_point["label"] in lm_to_rename.keys():
                control_point["label"] = lm_to_rename[control_point["label"]]
            new_control_Points.append(control_point)

    data["markups"][0]["controlPoints"] = new_control_Points

    with open(fiducial_file, 'w') as f:
        json.dump(data, f, indent=4)
        

        # print(control_point["label"])

        # lm_ph_coord = np.array([markup["position"][2],markup["position"][1],markup["position"][0]])
        # for i in range(self.dim):
        #     lm_coord = (lm_ph_coord+ abs(self.origins[i]))/self.spacings[i]
        #     lm_coord = lm_coord.astype(int)
        #     self.dim_landmarks[i][markup["label"]] = lm_coord

    # self.original_dim_landmarks = copy.deepcopy(self.dim_landmarks)



# for file in fiducial_files:
#     fin = open(file, "rt")
#     #read file contents to string
#     data = fin.read()
#     #replace all occurrences of the required string
#     # data = data.replace("U3", "UL3")
#     # data = data.replace("U3A", "UL3A")

#     #close the input file
#     fin.close()
#     #open the input file in write mode
#     fin = open(file, "wt")
#     #overrite the input file with the resulting data
#     fin.write(data)
#     #close the file
#     fin.close()


