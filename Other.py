import argparse
import glob
import sys
import os
import fileinput
import pandas as pd


def main(args):

    print("Reading folder : ", args.input_dir)


    patients = {}
    		
    normpath = os.path.normpath("/".join([args.input_dir, '**', '']))
    for img_fn in sorted(glob.iglob(normpath, recursive=True)):
        #  print(img_fn)
        basename = os.path.basename(img_fn)

        if True in [scan in img_fn for scan in [".fcsv",".mrk.json"]]:
            file_name = basename.split(".")[0]
            elements = file_name.split("_")
            patient = elements[0]
            if patient not in patients.keys():
                patients[patient] = {"dir": os.path.dirname(img_fn)}

            patients[patient]["fid"] = img_fn


    dfs = pd.read_excel(args.file)
    # print(dfs["L"])

    # print(patients)

    REPLACE = ['2','2A','3','3A','1A','1','6']
    Sides = ['L','R']
    for side in Sides:
        for key in dfs[side]:
            if key in patients.keys():
                # print(patients[key])
                file = patients[key]["fid"]
                fileToSearch  = file
                print('Reading file :', fileToSearch)

                tempFile = open( fileToSearch, 'r+' )
                for line in fileinput.input( fileToSearch ):
                    # if textToSearch in line :
                    #     print(textToSearch,'Found')
                    newline = line
                    for txt in REPLACE:
                        if 'U'+txt in line:
                            newline = line.replace( 'U'+txt, 'U'+side+txt )
                    tempFile.write(newline)
                        
                tempFile.close()
                # print('File done :', fileToSearch)






    # for patient,data in patients.items():
        
    #     file = data["fid"]


        # textToSearch = input( "> " )
        # textToReplace = input( "> " )
        # fileToSearch  = file
        # #fileToSearch = 'D:\dummy1.txt'

        # tempFile = open( fileToSearch, 'r+' )

        # for line in fileinput.input( fileToSearch ):
        #     if textToSearch in line :
        #         print(textToSearch,'Found')
        #     tempFile.write( line.replace( textToSearch, textToReplace ) )
        # tempFile.close()




if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='MD_reader', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    input_group = parser.add_argument_group('Input files')
    input_group.add_argument('-i','--input_dir', type=str, help='Input directory with 3D images',required=True)

    # output_params = parser.add_argument_group('Output parameters')
    # output_params.add_argument('-o','--out', type=str, help='Output directory', required=True)

    input_group.add_argument('-f', '--file', type=str, help='file', default='/Users/luciacev-admin/Desktop/Book1.xlsx')

    args = parser.parse_args()
    
    main(args)

# def CorrectCSV(filePath, Rcar = [" ", "-1"], Rlab = ["RGo_LGo", "RCo_LCo", "LCo_RCo", "LGo_RGo"]):
#     """
#     Remove all the unwanted parts of a fiducial file ".fcsv" :
#     - the spaces " "
#     - the dash ! "-1"
#     _ the labels in the list

#     Parameters
#     ----------
#     filePath
#      path of the .fcsv file 
#     """
#     file_data = []
#     with open(filePath, mode='r') as fcsv_file:
#         csv_reader = csv.reader(fcsv_file)
#         for row in csv_reader:
#             file_data.append(row)

#     with open(filePath, mode='w') as fcsv_file:
#         writer = csv.writer(fcsv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

#         for row in file_data:
#             keep = True
#             if "#" not in row[0]:
#                 for car in Rcar : row[11] = row[11].replace(car,"")
#                 if True in [label in row[11] for label in Rlab] : keep = False

#             if(keep):
#                 writer.writerow(row)