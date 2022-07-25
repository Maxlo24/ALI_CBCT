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



Scripts for Automatic Landmark Identification in CBCT

## Prerequisites

python 3.8.8 with the anaconda environment "ALI_environment.yml":

**Main librairies:**

> monai==0.7.0 \
> torch==1.10.1 \
> itk==5.2.1 \
> numpy==1.20.1 \
> simpleitk==2.1.1

# Running the code

## Using Docker
You can get the ALI CBCT docker image by running the folowing command lines.

**Informations**
- A ***test scan*** "MG_scan_test.nii.gz" is provided in the Data folder of the ALI_CBCT repositorie.
- If the prediction with the ***GPU is not working***, make sure you installed the NVIDIA Container Toolkit : 
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker

**Building using the DockerFile**

From the DockerFile directory:
```
docker pull dcbia/alicbct:latest
```

From the DockerFile directory:

```
docker build -t alicbct .
```

**Automatic landmark identification**
*Running on CPU*

```
docker run --rm --shm-size=5gb -v <Folder_with_the_scans_to_segment>:/app/data/scans -v <Folder_with_the_models_to_use>:/app/data/models  alicbct:latest python3 /app/ALI_CBCT/predict_landmarks.py
```
*Running on GPU*
```
docker run --rm --shm-size=5gb --gpus all -v <Folder_with_the_scans_to_segment>:/app/data/scans -v <Folder_with_the_models_to_use>:/app/data/models  alicbct:latest python3 /app/ALI_CBCT/predict_landmarks.py
```

**options/arguments**
- By default, only *Ba* and *S* landmarks are selected.
    To choose which structure to segment, you can use the following arguments:
    ```
    -lm Ba S
    ```
    <!-- To deactivate the merging step, you can use the following argument:
    ```
    -m False
    ``` -->

___





**Pre-process**

To run the preprocess to prepare the files and set them at the wanted spacing for the training:

For the Upper, Lower and Cranial base landmarks
```
python3 init_training_data_ULCB.py -i "path of the input folder with the scans and the fiducial list" -o "path of the output folder"
```

For the canine impaction landmarks
```
python3 init_training_data_Canine.py -i "path of the input folder with the scans and the fiducial list" -o "path of the output folder"
```


By defaul the spacing is set at 1 and 0.3 but we can change and add other spacing with the argument :
```
-sp x.xx x.xx
````
A contrast adjustment is also applied but can be turned off with 
```
-ch False
````
---




#Images

Environment, low resolution and high resolution:
![2environement_label_zoom](https://user-images.githubusercontent.com/46842010/159337231-0e79e134-a027-4987-ab44-edc2ad54d244.png)


Agent architecture:
![agent_label](https://user-images.githubusercontent.com/46842010/159341624-5d17e5a3-c4b7-4b93-bd7d-0b1348c7ad31.png)

Search steps of the agent to find one landmark:
![Search_3Steps_labeled](https://user-images.githubusercontent.com/46842010/159337300-ecb9e70e-7a65-45e1-96b1-490ad7286aa7.png)