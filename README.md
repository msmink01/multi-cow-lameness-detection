# Automatic Lameness Detection in Complex Multi-Cow Scenes

By Moniek Smink, Pratham Johari, and Talitha Aarnoutse (Group 16; DSAIT 4125 at TU-Delft) in collaboration with the University of Wisconsin-Madison School of Veterinary Medicine.

# Background & Motivation

Cattle husbandry, or the raising of cattle to produce diversified products for consumption, is a huge worldwide industry. The US alone houses more than 89 million cattle, creating an income of more than $80 billion every year<sup>1</sup>. 

One of the top challenges when raising cattle is cattle lameness, seen as abnormal movement, such as limping, and/or extreme posture, such as abnormal headbobbing or back arching. Lameness is a sign of pain or discomfort within the locomotor system of the animal and is often caused by bacterial infection or improper hoof trimming. Lameness is often measured on different scales. For the purposes of this report, we separate two types of cattle lameness: 

1. Clinical Lameness: when the animal is noticeably lame.
2. Subclinical Lameness: when the animal's lameness is subtle.

Please see examples of a clinical and subclinical animal in Figure 1. 

<h5>Figure 1: Examples of a clinically lame cow (left image; first cow: brown) and a subclinically lame cow (right image; third cow: brown).</h5>

<img src="./figures/Clinical.gif" alt="Clinically lame cow" width="300"> <img src="./figures/Subclinical2.gif" alt="Subclinically lame cow" width="300">

Lameness is a huge issue, impacting around 21% of all the world's cattle <sup>3</sup>. If left untreated, lameness is extremely painful and has severe impacts on the health and welfare of a cow. It also is a severe detriment to a farm's production, significantly lowering milk yields<sup>3</sup>. Lameness is often a sign of infection, which, if left untreated, can cause extremely high somatic cell counts (antibody cell counts) in the milk of a cow, also decreasing the quality of the milk. This is a multi-billion $/€ issue affecting hundreds of millions of animals and farmers.

What also makes cattle lameness such a significant issue is how challenging it is to manage. Clinical cases are easier to identify but are not always treatable<sup>3</sup>. Subclinical cases are almost always cheaply treatable, but are extremely difficult for untrained eyes to identify<sup>3</sup>. Ideally, a farmer would want to identify lameness in the subclinical stage because of the easier and cheaper treatment. On huge farms with thousands of cattle, spotting these subclinical cases early is incredibly difficult.

Thus, the cattle husbandry industry has turned to computer vision as a potential solution to automatically detect lameness in cattle. These systems must be portable to real-world environments, meaning that they must function with limited computational resources, without wifi, and in dirty environments. Furthermore, these systems must operate in close to real-time because, on thousand cow dairies, a system will not have time to process a backlog of footage before a new batch of cows arrive.

### Related Work

Several approaches exist, commercially and in the research community, that automatically detect lameness in cattle, but these approaches are often sensitive to inter- and intra- animal variability, such as breed of cow, color of cow, and environmental factors <sup>4,5,6</sup>. Furthermore, these systems, as far as we are aware, are always for clean, single-cow scenes, where one cow smoothly walks across the frames without human interference. Open-source approaches are usually created and evaluated in small farm environments with less than 200 cows. Thus, these approaches generalize poorly to huge farm environments where cows display typical cow behavior with and without human interference. Examples of these complex multi-cow scenes can be seen in Figure 2.

<h5>Figure 2: Examples of a complex multi-cow scenes: with human interference (left), with severe occlusion (all), and with typical cow behavior (right).</h5>

<img src="./figures/multi.gif" alt="Multi-cow scene example 1" width="300"> <img src="./figures/multi2.gif" alt="Multi-cow scene example 2" width="300"> <img src="./figures/snout.gif" alt="Multi-cow scene example 3" width="300">

Finally, none of the open-source approaches for automatic lameness detection use more advanced computer vision techniques such as end-to-end video action recognition <sup>7</sup>, or skeletal action recognition<sup>4,5,6</sup>, although other tasks such as cattle behavior recognition have explored such methods <sup>10</sup>. Existing approaches often employ a three stage approach. First, they localize the animal using foot detection, cow pose estimation, or cow segmentation with methods such as Faster-RCNN, T-LEAP, and Mask-RCNN. They then calculate hand-made features such as step length, head bob, and back arch coefficients. Finally they classify based on traditional discriminative classifiers such as support vector machines (SVMs), logistic regression, or decision trees<sup>4,5,6</sup>.

A classic end-to-end video action recognition method leveraged in this report is Inflated 3D networks (I3D). I3D is a two stream 3D convolutional neural network (CNN) that uses both RGB and optical flow frame inputs to classify human actions such as kicking and punching in videos <sup>7</sup>. Although not fully explored in this report, typical skeletal action recognition methods include spatial temporal graph convolutional networks (ST-GCNs) <sup>8</sup> and Pose C3Ds<sup>9</sup> which process spatiotemporal features using graph convolutions and 3D heatmap convolutions, respectively, to classify actions in videos.

### Research Questions

We aim to explore automatic lameness detection in complex, multi-cow scenes (shown in Figure 2) using more recent computer vision methods than existing approaches. Our research questions include:

1. How does an existing computer vision technique, such as end-to-end video action recognition, perform in a difficult task such as multi-cow lameness detection?
2. How can an automatic lameness detection system be made robust to inter- and intra- animal variability in a large production environment, while still being portable and efficient?

# Methods

## The Data

### Data Collection

### Data Annotation

## Methods & Intermediate Results

### Video Action Recognition

### Multi-Cow Localization + Classification

#### Pose Estimation + Tracking: Yolov8l-pose + BoT-SORT

<img src="./figures/YoloPreds.gif" alt="Example outputs of the yolo model and tracking layer" width="500">

#### Lameness Classification

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

