# Automatic Landmark Identification in 3D Cone-Beam Computed Tomography scans

Authors:
Maxime Gillot
Baptiste Baquero
Antonio Ruellas
Marcela Gurgel
Najla Al Turkestani
Elizabeth Biggs
Marilia Yatabe
Jonas Bianchi
Lucia Cevidanes
Juan Carlos Prieto

We propose a novel approach that reformulates landmark detection as a classification problem through a virtual agent placed inside a 3D Cone-Beam Computed Tomography (CBCT) scan. This agent is trained to navigate in a multi-scale volumetric space to reach the estimated landmark position.

Landmark placed in the CBCT:
![LM_SELECTION_Trans](https://user-images.githubusercontent.com/46842010/159336503-827d70d5-2212-4dea-8ccc-46fc420be2e2.png)

Environment, low resolution and high resolution:
![2environement_label_zoom](https://user-images.githubusercontent.com/46842010/159337231-0e79e134-a027-4987-ab44-edc2ad54d244.png)

Agent architecture:
![agent_label](https://user-images.githubusercontent.com/46842010/159341624-5d17e5a3-c4b7-4b93-bd7d-0b1348c7ad31.png)

Search steps of the agent to find one landmark:
![Search_3Steps_labeled](https://user-images.githubusercontent.com/46842010/159337300-ecb9e70e-7a65-45e1-96b1-490ad7286aa7.png)



Scripts for Automatic Landmark Identification in CBCT

## Prerequisites

python 3.8.8 with the anaconda environment "ALI_environment.yml":

**Main librairies:**

> monai==0.7.0 \
> torch==1.10.1 \
> itk==5.2.1 \
> numpy==1.20.1 \
> simpleitk==2.1.1

## Running the code

**Pre-process**

To run the preprocess to organise the files and set them at the wanted spacing:

```
python3 init_training_data.py -i "path of the input folder with the scans and the segs" -o "path of the output folder"
```
By defaul the spacing is set at 0.5 but we can change and add other spacing with the argument :
```
-sp 0.X1 0.X2 ... 0.Xn 
````

To run the preprocess to correct the image contrast and fill the holes in the segmentations , run the folowing command line:

```
python3 correct_file.py -i "path of the input folder with the scans and the segs" -o "path of the output folder"
```
