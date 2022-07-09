# README
### Make sure to cite our paper: 
 S. Lamczyk, K. Ampofo, B. Salahshour, M. Cetin, and K. Iftekharuddin, “SURFGenerator: Generative Adversarial Network Modeling for Synthetic Flooding Video Generation” IJCNN at IEEE WCCI 2022, 2022


## Masker
### Requirements
- segmentation_models
- tensorflow
- glob
- natsort
- opencv-python
- tqdm

Utilizing our masker network is pretty straightforward. No downloads are required for using our weights. However, you can access our data here and arrange them in the directory structure indicated in masker/train.py if you would like to train our model:  
Download the different masks here: https://drive.google.com/file/d/1X3Ru3hXB_tsQ6dTv-VUIVGKS_JYbUSfl/view?usp=sharing

Download the images here: https://drive.google.com/file/d/1xocJcazHnOmPt_keY_EN38hAOeugzgdL/view?usp=sharing

#### Inference
- Firstly, break down your desired input video into its frames and place the frames(numbered from 0 to whatever) in the inputSequence folder.
- Secondly, create a reference mask(numbered 0) and place it in the outputMasks directory. The mask should be placed in the outputMasks directory. Remember that it should be 1 where you want flooding and 0 where you do not want flooding.
- Third, run the maskerinference.py file to get the output sequence in the outputSequence directory. Now you can take this sequence and feed it to vid2vid.

## Painter
This link provides the checkpoint to test our model:  
https://drive.google.com/file/d/1lGVy3CS3sEO5qJ_Zd3IeGJpH_-MUTAwl/view?usp=sharing  
The checkpoints folder should be inside vid2vid such as painter/checkpoints

This provides the data that we used to train vid2vid:  
https://drive.google.com/file/d/1aY4d_COYJjtguGvekX8t-E1wtRRr8ltr/view?usp=sharing  
This should be setup as painter/datasets/...

This is the command we run from the painter directory to perform inference in data located at painter/datasets/Cityscapes/test_A  
```sh
python test.py --name 512 --input_nc 3 --loadSize 512 --n_scales_spatial 2 --use_real_img
```
A corresponding test_B folder is also required. This test_B folder can be composed of really anything as long as it is the same length as the amount of images in test_A. However, an initial image is required for vid2vid to start working. We provide with this repository a copy of pix2pixHD for creating the initial image. Before utilizing pix2pix, make sure to unzip this folder into the pix2pix directory. The structure should be pix2pix/synth

https://drive.google.com/file/d/1jAyCJ2q97eSh6aVYMcxsCbYx8vYbLXCR/view?usp=sharing

You can perform inference on your image as follows. We provide an example image and output

```sh
python pix2pix.py \
  --mode test \
  --output_dir results \
  --input_dir test \
  --checkpoint synth
  ```
