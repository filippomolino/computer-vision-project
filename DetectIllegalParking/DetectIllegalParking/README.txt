
Link to Google Drive: https://drive.google.com/drive/folders/1zTBQw1RvXaAKjKxeC_rIPgF1ZWj41Hjk?usp=share_link 
To upload dataset and necessary files to Google Collab, you need to first do (REQUIRED STEP):

1. Go to Google Drive 
2. In the “Shared with me” section, right click the shared folder
3. Click “Add shortcut to Drive” and place it somewhere in “My Drive”
4. Then you can access it in Collab from the first code cell 

Then in the second code cell, we make the yolo_car_detection folder as a python package so that we can refer to it in upcoming cells. (REQUIRED STEP)

Third cell which begins with the comment: Object Detection and 3D Bounding Boxes, is not effecting the main workflow of the pipeline. It is included just for visualizing our predictions for Object Detection task. You can run it for demonstration purposes, or skip it.  (OPTIONAL STEP)

Fourth cell stands for Inference Code for Segmentation task and it generates Image Masks which is going to be used in the main pipeline, thus it must be compiled. (REQUIRED STEP)

On the Fifth cell we execute the Inference Code in the previous cell. It makes a new folder under the dataset folder, namely masks, and it stores the generated Masks as well as the Overlays of the given image.  In the arguments:
1. It calls for seg_cpu.pt which is the segmentation model that executes on CPU. 
2. As for the input there are two ways of passing the images:

	2.1. For executing the model on the whole dataset, use: 
		/content/dataset/images/*.png  argument. This will generate segmentation for every image under 		the dataset. (ADVISED FOR FIRST TIME EXECUTION)

	2.2. For executing the model on one image, use:
		/content/dataset/images/frankfurt_000001_077434_leftImg8bit.png. This image path is a 		placeholder, you can replace it with any other image. This will generate segmentation only for the 		specified image.  (OPTIONAL, FOR TESTING PURPOSES)

3. The model saves the results into specified destination


Last cell is the MAIN PIPELINE. It gets the segmentation masks for the images, then does object detection and depth estimation on the images, continues with computing bottom surface area of the 3D bounding box, then calculates the overlapping percentage. This cell is designed to generate illegal parking results for every image under the dataset, alongside with the corresponding generated masks:

IMAGE_SOURCE = “/content/dataset/images” is set to fetch every image, this can be changed to point to a specific image with the format of “/content/dataset/images/my.png”. 

MASK_SOURCE = “/content/dataset/masks” is set to fetch every segmentation mask, this can be changed to point a specific mask with the format of “/content/dataset/masks/my_mask.png”. 


After executing the last cell, it will create a new folder namely “results” where you can see the generated outputs with their respective name. 

#########
Note that there is an additional file namely segmentation_model_training.py, this file is not being used in the pipeline at all, it is included just in case for examining how we have trained our segmentaiton model.