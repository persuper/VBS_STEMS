# VBS_STEMS
Phenotyping api for CT images of maize stems 
# PHENOTYPING API FOR VASCULAR BUNDLES (VBS) #


2022 V5

**Note**: 
1. The code in this repository was tested with torch 1.11
2. Deeplabv3 for semantic segmentation
3. Adaptive watershed for the identification of vbs
4. Layer quantification for the cross-section CT
5. Mechanical properties to do
6. FEM modeling to do
7. Must install mmdet, mmseg
8. The models will be provided

## REQUIREMENTS ##
    torch==1.11.0+cu115
    numpy==1.22.4
    scipy==1.7.3
    pycocotools==2.0.4
    opencv-python==4.5.4.60
    six==1.16.0
    terminaltables==3.1.10
    matplotlib==3.5.1
    typing==3.7.4.3
    sklearn==0.0
    scikit-learn==1.1.2
    imagecorruptions==1.1.2
    pillow==9.2.0
    requests==2.28.1
    seaborn==0.11.2
    onnxruntime==1.12.1
    cityscapesscripts==2.2.0
    timm==0.6.7
    prettytable==3.3.0
    packaging==21.3
    pandas==1.3.5
    exifread==3.0.0
    flask==2.1.3
    imutils==0.5.4
    scikit-image==0.19.3

## TRAITS
    
    Functional Zone	Trait	Description	Unit
    Cross-section region	
        SZ_A	The area of the cross-section zone	mm2
        SZ_P	The perimeter of the cross-section zone	mm
        SZ_LA	The long axis length of the stem cross-section	mm
        SZ_SA	The short axis length of the stem cross-section	mm
        SZ_CA	The convex area of the stem cross-section	mm2
        SZ_CCA	The circumcircle area of the stem cross-section	mm2
        SZ_CAR	The convex area ratio: SZ_CA / SZ_A	-
        SZ_LWR	The length-width ratio of the stem cross-section	-
    Epidermis region	
        EZ_A	The area of the epidermis zone	mm2
        EZ_T	The thickness of the epidermis zone	mm
    Periphery region	
        PZ_A	The area of the periphery zone	mm2
        PZ_T	The thickness of the periphery zone	mm
        PZ_VB_N	The number of vascular bundles in the periphery zone	-
        PZ_VB_A	The total area of vascular bundles in the periphery zone	mm2
        PZ_VB_D	The density of vascular bundles in the periphery zone	number/mm2
        PZ_VB_CA	The convex area of vascular bundles in the periphery zone	mm2
        PZ_VB_CAR	The convex area ratio of vascular bundles in the periphery zone: PZ_VB_CA/ PZ_VB_A	-
    Inner region	
        IZ_A	The area of the inner zone	mm2
        IZ_T	The thickness of the inner zone	mm
        IZ_VB_N	The number of vascular bundles in the inner zone	-
        IZ_VB_A	The total area of vascular bundles in the inner zone	mm2
        IZ_VB_D	The density of vascular bundles in the inner zone	number/mm2
        IZ_VB_CA	The convex area of vascular bundles in the inner zone	mm2
        IZ_VB_CAR	The convex area ratio of vascular bundles in the inner zone: IZ_VB_CA/ IZ_VB_A	-
    Total vascular bundle	
        VB_N	The total number of vascular bundles	-
        VB_A	The total area of vascular bundles of the stem	mm2
        VB_Aave	The average area of vascular bundles	mm2
        VB_Pave	The average perimeter of vascular bundles	mm
        VB_Laave	The average long axis length of vascular bundles	mm
        VB_Saave	The average short axis length of vascular bundles	mm
        VB_Caave	The average convex area of vascular bundles	mm2
        VB_CCAave	The average circumcircle area of vascular bundles	mm2
        VB_CAR	The convex area ratio of vascular bundles: VB_CA/ VB_A	-
        VB_LWR	The length-width ratio of vascular bundles	-
        ARIVB	Area ratio of individual vascular bundles	-
        SRVB	Separation ratio of vascular bundles	-
        VB_D	The density of vascular bundles: total number of vascular bundles /area of the cross-section zone 	number/mm2
        VB_AreaRatio	The total area of vascular bundles/ the area of the cross-section zone	-

## Citation
If you find the code in this repository useful for your research consider citing it.


Jianjun Du, Ying Zhang, Xianju Lu, Minggang Zhang, Jinglu Wang, Shengjin Liao, Xinyu Guo, Chunjiang Zhao, (2022) A deep learning-integrated phenotyping pipeline for vascular bundle phenotypes and its application in evaluating sap flow in the maize stem. Crop Journal, https://doi.org/10.1016/j.cj.2022.04.012.

