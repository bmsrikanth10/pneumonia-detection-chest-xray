			Student Author: Srikanth Banagere Manjunatha sxb5973@psu.edu

				  Title: Pneumonia Detection using X-Net




The final_submission folder contains following folders and the content of these files are as follows:

1) Proposal:
	This folder contains the Proposal document which was submitted in the initial phase of the course project.

2) Related_works:
	This folder contains the related works (in pdf formats with the same prompted names from the internet)
	referred for this course project.

3) Srikanth_Banagere_Manjunatha_final_presentation:
	This folder contains the final presentation slides and the edited video which was presented during the final
	phase of the project. This folder also contains the final report in the pdf format as a backup.

4) Srikanth_Banagere_Manjunatha_midterm_progress:
	This folder contains the mid term progress presentation slides and the video which was presented during the
	mid-phase of the project. This folder also contains the mid term report in the pdf format which was submitted
	during the mid-phase of the project.

5) Term_proj:
	This folder contains the data set required for this project. This folder contains the following subfolders:
		i)    aug_dir:
			This contains all the augmented training images in two folders "Normal" and "Pneumonia".
		ii)   test:
			This contains all the test images in two folders "Normal" and "Pneumonia".
		iii)  train:
			This contains all the unaugmented original training images in two folders "Normal" and
			"Pneumonia".
		iv)   val:
			This contains all the validation images in two folders "Normal" and "Pneumonia".

6) weights:
	This folder contains the following:
		i)    subfolder "with": which contains all the filter visualization images for Baseline model with
			data augmentation.
		ii)   subfolder "without": which contains all the filter visualization images for Baseline model
			without data augmentation.
		iii)  subfolder "xnet": which contains all the filter visualization images for X-Net model.
		iv)   plot_auc.py: code to generate AUC curve and precision Vs recall curve.
		v)    tsne_vs.py: code to visualize the TSNE plot after each convolution layer and to understand the
			separation between each class as we go deeper.
		vi)   weight_visualize.m: code to visualize the filters after each convolution layer as we go deeper
			for Baseline model.
		vii)  weight_visualize_xnet.m: code to visualize the filters after each convolution layer as we go
			deeper for X-Net model.
		viii) baseline_model_without.h5: model information saved in h5 format.
		ix)   baseline_model_with.h5: model information saved in h5 format.
		x)    xnet_model_v1_10.h5: model information saved in h5 format.

7) five_splits:
	This folder contains five different splits of the training and test images for the experiments.

8) baseline_code_with_aug.py:
	code to train the Baseline model with data augmentation. 

9) baseline_code_without_aug:
	code to train the Baseline model without data augmentation.

10) data_gen_for_aug:
	code to generate the augmented data.

11) kmeans_code:
	code to evaluate the unsupervised learning approach (K-means) on the data.

12) knn_data_study_code:
	code to find the closest training image for each of the test image. We perform data study based on the L1 and
	L2 distance metric for each test image and the closest training image.

13) Project_Final_Report_PR:
	The final Term Project Report containing all the details and results about the course project.

14) xnet_code:
	code to train the X-Net model. 

16) readme.txt: A brief information about the content of the folder final_submission.












		