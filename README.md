# Automatic Lameness Detection in Complex Multi-Cow Scenes

By Moniek Smink, Pratham Johari, and Talitha Aarnoutse (Group 16; DSAIT 4125 at TU-Delft) in collaboration with the University of Wisconsin-Madison School of Veterinary Medicine.

# Background & Motivation

Cattle husbandry, or the raising of cattle to produce diversified products for consumption, is a huge worldwide industry. The US alone houses more than 89 million cattle, creating an income of more than $80 billion every year<sup>1</sup>. 

After reproduction and mastitis issues, the third most import dairy production health problem is cattle lameness, seen as abnormal movement and/or extreme posture, such as limping, abnormal headbobbing, or back arching. Lameness is a sign of pain or discomfort within the locomotor system of the animal and is often caused by bacterial infection or improper hoof trimming. Lameness is measured in different ways, including 3, 4, and 9 point scales. For the purposes of this report, we separate two types of cattle lameness: 

1. Clinical Lameness: when the animal is noticeably lame.
2. Subclinical Lameness: when the animal's lameness is subtle.

Please see examples of a clinical and subclinical animal in Figure 1. 

<h5>Figure 1: Examples of a clinically lame cow (left image; first cow: brown) and a subclinically lame cow (right image; third cow: brown).</h5>

<img src="./figures/Clinical.gif" alt="Clinically lame cow" width="300"> <img src="./figures/Subclinical2.gif" alt="Subclinically lame cow" width="300">

Lameness is a huge issue, impacting around 21% of all the world's cattle<sup>3</sup>. If left untreated, lameness is extremely painful and has severe impacts on the health and welfare of a cow. It also is a severe detriment to a farm's production, significantly lowering milk yields<sup>3</sup>. Lameness is often a sign of infection, which, if left untreated, can cause extremely high somatic cell counts (antibody cell counts) in the milk of a cow, also decreasing the quality of the milk. This is a multi-billion $/€ issue affecting hundreds of millions of animals and farmers.

What also makes cattle lameness such a significant issue is how challenging it is to manage. Clinical cases are easier to identify but are not always treatable<sup>3</sup>. Subclinical cases are almost always cheaply treatable, but are extremely difficult for untrained eyes to identify<sup>3</sup>. Ideally, a farmer would want to identify lameness in the subclinical stage because of the easier and cheaper treatment. On huge farms with thousands of cattle, spotting these subclinical cases early is incredibly difficult.

Thus, the cattle husbandry industry has turned to computer vision as a potential solution to automatically detect lameness in cattle. These systems must be portable to real-world environments, meaning that they must function with limited computational resources, without wifi, and in dirty environments. Furthermore, these systems must operate in close to real-time because, on thousand cow dairies, a system will not have time to process a backlog of footage before a new batch of cows arrive.

### Related Work

Several approaches exist, commercially and in the research community, that automatically detect lameness in cattle, but these approaches are often sensitive to inter- and intra- animal variability, such as breed of cow, color of cow, and environmental factors <sup>4,5,6</sup>. Furthermore, these systems, as far as we are aware, are always for clean, single-cow scenes, where one cow smoothly walks across the frames without human interference. Open-source approaches are usually created and evaluated in small farm environments with less than 200 cows. Thus, these approaches generalize poorly to huge farm environments where cows display typical cow behavior with and without human interference. Examples of these complex multi-cow scenes can be seen in Figure 2.

<h5>Figure 2: Examples of a complex multi-cow scenes: with human interference (left), with severe occlusion (all), and with typical cow behavior (right).</h5>

<img src="./figures/multi.gif" alt="Multi-cow scene example 1" width="300"> <img src="./figures/multi2.gif" alt="Multi-cow scene example 2" width="300"> <img src="./figures/snout.gif" alt="Multi-cow scene example 3" width="300">

Finally, none of the open-source approaches for automatic lameness detection use more advanced computer vision techniques such as end-to-end video action recognition <sup>7</sup>, or skeletal action recognition<sup>4,5,6</sup>, although other tasks such as cattle behavior recognition have explored such methods <sup>10</sup>. Existing approaches often employ a three stage approach. First, they localize the animal using foot detection, cow pose estimation, or cow segmentation with methods such as Faster-RCNN, T-LEAP, and Mask-RCNN. They then calculate hand-made features such as step length, head bob, and back arch coefficients. Finally they classify based on traditional discriminative classifiers such as support vector machines (SVMs), logistic regression, or decision trees<sup>4,5,6</sup>.

A classic end-to-end video action recognition method leveraged in this report is Inflated 3D networks (I3D). I3D is a two stream 3D convolutional neural network (CNN) that uses both RGB and optical flow frame inputs to classify human actions such as kicking and punching in videos<sup>7</sup>. Although not fully explored in this report, typical skeletal action recognition methods include spatial temporal graph convolutional networks (ST-GCNs) <sup>8</sup> and Pose C3Ds<sup>9</sup> which process spatiotemporal features using graph convolutions and 3D heatmap convolutions, respectively, to classify actions in videos. Finally, the temporal component needed to identify lameness makes attention mechanisms and RNN mechanisms particularly promising, as both of these methods have ways to combine temporal contexts to reason over time.

### Research Questions

We aim to explore automatic lameness detection in complex, multi-cow scenes (shown in Figure 2) using more recent computer vision methods than existing approaches. Our research questions include:

1. How does an existing computer vision technique, such as end-to-end video action recognition, perform in a difficult task such as multi-cow lameness detection?
2. How can an automatic lameness detection system be made robust to inter- and intra- animal variability in a large production environment, while still being portable and efficient?

# Methods & Intermediate Results

## The Data

To answer our research questions, it was clear that we would need the help of experienced professionals, not only to get access to complex, multi-cow scene videos, but also to help verify lameness labels. Associate Professor and veterinarian, Dr. Dörte Döpfer, in the Food Animal Production Medicine (FAPM) department at UW-Madison's School of Veterinary Medicine (SVM), recorded more than six hours of footage and graciously allowed us access to it with the agreement that this data was highly private and any resulting labeling or work would be under her supervision, but we would be credited. The six hours of footage were recorded with informed consent using GoPro cameras at a private midwestern dairy with more than 9 thousand cows of varying ages, breeds, and colorings. 

The data was annotated by us in three different ways to be used in three different methods all of which is explained below. Labels related to lameness were verified by Dr. Döpfer.

## Approaches & Intermediate Results

## 1. Video Action Recognition

To the best of our knowledge, no one in existing literature has tried end-to-end video action recognition on cattle lameness before. Thus, we decided to try this.

### Data Annotation

Subsets of the 6 hours of source footage were randomly selected for processing and were split into over one thousand 5 second clips. These 5 second clips were then either pseudo labeled by us and then verified by Dr. Döpfer, or directly labeled by Dr. Döpfer without our assistance. This yielded 1,015 clips with a 'Not Lame' (78.23%), 'Subclinically Lame' (13.69%), or 'Clinically Lame' (8.08%) label. It was decided that if there was a single cow in the clip that was lame, the whole clip would be labeled as such, with a worst label priority rule. These clips were then split into train (80%), validation (10%), and test (10%) sets to be used to train and evaluate our video action recognition models. Since the distribution of the different labels was significantly unbalanced, a balanced version of the training dataset was also created where the two lame labels were oversampled (essentially tripled) to form a label distribution of 38.63%, 36.10%, and 25.27%. The 'Not Lame' label was not undersampled to still allow the model to learn animal variability.

### Model & Intermediate Results

For this experiment, we leveraged I3D<sup>7</sup>, a standard video action recognition model pretrained on Kinetics-400 which takes 32 224x224 frames as input. We finetuned the model for 10 epochs using stochastic gradient descent with random cropping and horizontal flipping on the unbalanced and balanced datasets described above. The resulting top1 accuracy and average top1 accuracy across the three classes is shown in Table 1.

<h5>Table 1: Top1 accuracy and average top1 accuracy of all classes of the finetuned I3D model on our custom cattle lameness dataset.</h5>

| Dataset   | Top1 Accuracy† | Mean Class Top1 Accuracy  |
|-----------|------------|-----------------------|
| Raw       | 71.28      | 37.41                 |
| Balanced  | 47.52      | 42.42                 |

<h6>† Top1 Accuracy refers to the percentage of samples for which the top predicted class is the correct label.</h6>

### Discussion & Next Steps

We see that end-to-end video action recognition has a lot of trouble with spotting lameness. I3D is pretrained on human actions which are often obvious from a single frame or a few frames, thus its 32 resized frame sampling strategy is effective. In our case, fine-grained (frame-by-frame) temporal relationships are necessary to identify lameness, which I3D's 32 resized frame sampling strategy can't deal with effectively. Furthermore, I3D must learn to deal with a lot of variability in each scene including cow positions, cow colorings, and cow occlusion. Perhaps controlling the sampling strategy to be more fine-grained could improve I3D's performance, but we simply don't have enough data to teach an I3D model how to handle this fine-grained temporal information while also being robust to scene variabilities. Thus, we look into approaches that abstract away the scene variabilities to hopefully focus on the important temporal information that can signify lamenesse in a cow. These dimensionality reduction approaches also have the added benefit of likely being much faster than I3D.

## 2. Multi-Cow Localization + Classification

In order to abstract away scene variabilities such as coloring, extraneous behavior, occlusion, and surroundings, we leveraged a two step approach. We first localize and track important features of a cow over time and then, based on that sequence of tracked features, predict lameness for that localized cow.

### Pose Estimation + Tracking

We first localize a cow using pose estimation with keypoints. This step removes any extraneous information about the surroundings of the cow and allows the downstream classification model to only focus on a set of features important to the lameness classification. Then, we track these features across the input frames, allowing the downstream classification model to learn fine-grained temporal relationships between the extracted features to hopefully classify lameness.

#### Data Annotation

We randomly select 1,015 frames from our source footage and label four keypoints with optional occlusion or out-of-frame flags per cow present in the frame using CVAT. The four keypoints we chose to label were:

1. Right Front (RF) foot
2. Right Rear (RR) foot
3. Left Rear (LR) foot
4. Left Front (LF) foot

We chose to label keypoints in this way to deal with occlusion and to make processing more efficient. Following previously defined 10-point or 17-point cow keypoint schemas would flood the classification model with potentially unecessary keypoints that are not usually available in complex multi-cow scenes. This would make our keypoint model and the downstream classification model more unstable and inefficient, thus we decided to only use the four feet keypoints in our approach. An example frame labeled by us along with the mentioned 10 and 17-point previous keypoint schemas are shown in Figure 3.

<h5>Figure 3: Example frame of our multi-cow 4-point keypoint labeling (left) and the 10-point (middle) and 17-point (right) cow keypoint schemas proposed in previous works.</h5>

<img src="./figures/KeypointLabeling.png" alt="Sample keypoint labeling frame" height="200"> <img src="./figures/KeypointSchemas.png" alt="Previously proposed cow keypoint schemas" height="200">

After removing frames with no animals present, 972 keypoint frames were then split into a train and validation set with an 80-20 split to be used to train the pose estimation model.

#### Model & Intermediate Results

##### YoloV8L-Pose

In order to accomplish multi-object pose estimation, we leveraged the YoloV8-Large-Pose model which takes a 640x640 frame and returns keypoints for each detected object. TODO: expand yolo model, how does it predict? Customized backbone, PAN-FPN neck, and pose estimation head blah blah. Uses a multi-part loss function that combines Complete Intersection over Union (CIoU) loss for the bounding boxes, Binary Cross-Entropy (BCE) loss for the objectness score, BCE loss for multi-class classification, and MSE loss for regressing the keypoint positions. We chose to use the Yolov8L-Pose model because of its competitive performance, extremely fast inference on edge devices, and ease of use. 

Our YoloV8L-Pose model was finetuned on our keypoint dataset described above for 300 epochs total, split into one training run for 200 epochs, and another for 100 epochs when we saw that pose validation loss had not yet converged. Training curves for this training can be seen in Figure 4. The final precision-recall curves for the bounding box and pose estimation can be seen in Figure 5. And the final mean Average Precisions (mAPs) for both bounding boxes and pose estimation can be seen in Table 2.

<h5>Figure 4: Training curves (top) and pose Precision-Recall curve (bottom) of the YoloV8L-Pose model on our custom multi-cow 4-point keypoint dataset.</h5>

<img src="./figures/TrainingCurves.png" alt="Training curves" height="250">

<img src="./figures/PosePR_curve.png" alt="Pose PR Curve" height="250">

<h5>Table 2: Mean Average Precisions (mAPs) for different Intersection over Union (IoU) thresholds for the final YoloV8L-Pose model trained on our custom multi-cow 4-point keypoint dataset.</h5>

| Type of Prediction   | mAP@0.5† | mAP@0.5:0.95*  |
|-----------|------------|-----------------------|
| Pose       | 48.49      | 31.96                 |
| Box  | 43.92      | 19.64                 |

<h6>† mAP@0.5: Mean Average Precision at IoU threshold of 0.5. Indicates how well the model detects objects/keypoints with a reasonable overlap.</h6>
<h6>* mAP@0.5:0.95: Mean Average Precision averaged over IoU thresholds from 0.5 to 0.95 (in steps of 0.05). A more stringent and comprehensive metric for model performance.</h6>

##### BoT-SORT

On top of the yolo model, we used a multi-object tracking algorithm called BoT-SORT to automatically assign a tracking id to each detected set of keypoints, based on previous frames' detections. BoT-SORT is an advanced algorithm that combines appearance features, motion prediction using Kalman filtering, history-detection matching using the Hungarian algorithm, and introduces Camera Motion Compensation (CMC) and appearance embedding (ReID) matching tricks. We chose this particular tracking layer because of its robustness to occlusion and real-time inference. Due to our limited time, we were not able to quantitatively evaluate the BoT-SORT tracking algorithm, but an overall qualitative example of how the final yolo + tracking pipeline functioned can be seen in Figure 6.

<h5>Figure 6: Example YoloV8L-pose keypoint estimation and BoT-SORT tracking (seen in 'id' field in text) for a short real-world clip.</h5>

<img src="./figures/YoloPreds.gif" alt="Example outputs of the yolo model and tracking layer" width="500">

#### Discussion

### Lameness Classification

#### Data Annotation

#### Model & Intermediate Results

#### Discussion

### Other Tried Methods

## Results

<img src="./figures/Final2.gif" alt="Final example outputs of our final pipeline" width="700">

## Conclusions & Future Work

### Future Work

Thank you for your attention.

<img src="./figures/snout.gif" alt="Multi-cow scene example 3" width="300">

## Poster

![Poster](./Poster.svg)

### Acknowledgements

We would like to thank the researchers and veterinarians at UW-Madison and the farmers and herds(wo)men who helped us with this project.

### References
1. Cattle & Beef - Sector at a glance (2025) Cattle & Beef - Sector at a Glance | USDA Economic Research Service. Available at: https://www.ers.usda.gov/topics/animal-products/cattle-beef/sector-at-a-glance.
2.
3. J.N. Huxley, Impact of lameness and claw lesions in cows on health and production, Livestock Science, Volume 156, Issues 1–3, 2013, Pages 64-70, ISSN 1871-1413, https://doi.org/10.1016/j.livsci.2013.06.012.
4. Myint, B.B., Onizuka, T., Tin, P. et al. Development of a real-time cattle lameness detection system using a single side-view camera. Sci Rep 14, 13734 (2024). https://doi.org/10.1038/s41598-024-64664-7
5. X. Kang, X.D. Zhang, G. Liu, Accurate detection of lameness in dairy cattle with computer vision: A new and individualized detection strategy based on the analysis of the supporting phase, Journal of Dairy Science, Volume 103, Issue 11, 2020, Pages 10628-10638, ISSN 0022-0302, https://doi.org/10.3168/jds.2020-18288.
6. Helena Russello, Rik van der Tol, Menno Holzhauer, Eldert J. van Henten, Gert Kootstra, Video-based automatic lameness detection of dairy cows using pose estimation and multiple locomotion traits, Computers and Electronics in Agriculture, Volume 223, 2024, 109040, ISSN 0168-1699, https://doi.org/10.1016/j.compag.2024.109040.
7. Yifan Peng, Jaesong Lee, & Shinji Watanabe. (2023). I3D: Transformer architectures with input-dependent dynamic depth for speech recognition.
8. Yan, S., Xiong, Y., & Lin, D. (2018). Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition. Proceedings of the AAAI Conference on Artificial Intelligence, 32(1). https://doi.org/10.1609/aaai.v32i1.12328
9. Haodong Duan, Yue Zhao, Kai Chen, Dahua Lin, & Bo Dai. (2022). Revisiting Skeleton-based Action Recognition.
10. Yongliang Qiao, Yangyang Guo, Keping Yu, Dongjian He. (2022) C3D-ConvLSTM based cow behaviour classification using video data for precision livestock farming, Computers and Electronics in Agriculture, Volume 193, 106650, ISSN 0168-1699, https://doi.org/10.1016/j.compag.2021.106650.

