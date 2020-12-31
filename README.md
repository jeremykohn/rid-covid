# Radiological Image Data for Clinical Open-source Viral Infection Diagnosis (RID-COVID)

A list of software projects, datasets, and other publicly available resources for diagnosing COVID-19 based on clinical images such as X-rays and CT scans.

For now, the goal is to help people find what already exists. Eventually this effort may develop into a project to produce new diagnostic tools suitable for clinical use.


## Image Data

- [ieee8023/covid-chestxray-dataset](https://github.com/ieee8023/covid-chestxray-dataset)
  - Dataset description: We are building a database of COVID-19 cases with chest X-ray or CT images. We are looking for COVID-19 cases as well as MERS, SARS, and ARDS.
    - Related research paper: [On the limits of cross-domain generalization in automated X-ray prediction](https://arxiv.org/abs/2002.02497) (Joseph Paul Cohen; Mohammad Hashir; Rupert Brooks; Hadrien Bertrand Mila, Université de Montréal)
  - License: (Some images in this dataset are licensed under CC BY, CC BY-NC-SA, CC BY-NC-ND, or Apache 2.0; for many other images, the license is unknown)

- [ml-workgroup/covid-19-image-repository](https://github.com/ml-workgroup/covid-19-image-repository)
  - Dataset description: This project aims to create an anonymized data set of COVID-19 cases with a focus on radiological imaging. This includes images with extensive metadata, such as admission-, ICU-, laboratory-, and patient master-data.
  - License: [Creative Commons CC BY 3.0](https://github.com/ml-workgroup/covid-19-image-repository/blob/master/LICENSE)

- [UCSD-AI4H/COVID-CT](https://github.com/UCSD-AI4H/COVID-CT)
  - Dataset description: The COVID-CT-Dataset has CT images containing clinical findings of COVID-19. We are continuously adding new COVID CT images and we would like to invite the community to contribute COVID CTs as well.
    - Related research paper: [COVID-CT-Dataset: A CT Scan Dataset about COVID-19](https://arxiv.org/abs/2003.13865) (Jinyu Zhao, UC San Diego; Yichen Zhang, UC San Diego; Xuehai He, UC San Diego; Pengtao Xie, UC San Diego, Petuum Inc)

- [agchung/Figure1-COVID-chestxray-dataset](https://github.com/agchung/Figure1-COVID-chestxray-dataset)
  - Dataset description: Figure 1 COVID-19 Chest X-ray Dataset Initiative. We are building this dataset as a part of the COVIDx dataset to enhance our models for COVID-19 detection. Please see the main [COVID-Net](https://github.com/lindawangg/COVID-Net) repo for details on data extraction and instructions for creating the full COVIDx dataset.

- [coyotespike/covid19-images](https://github.com/coyotespike/covid19-images)
  - Dataset description: A large collection of COVID-19 radiology imaging datasets for use in machine learning.

- [CORONACASES.ORG](https://coronacases.org/)
  - Dataset description: This community is for health professionals to share confirmed cases of coronavirus. We review posts in order to assure anonymization.

- [COVID-19 BSTI Imaging Database](https://www.bsti.org.uk/training-and-education/covid-19-bsti-imaging-database/)
  - Dataset description: The British Society of Thoracic Imaging (BSTI), in conjunction with Cimar UK’s Imaging Cloud Technology (cimar.co.uk), have designed, built and deployed a simple, free to use, anonymised and encrypted online portal to upload and refer imaging of patients with either confirmed or suspected COVID-19. From these cases, BSTI hope to provide an imaging database of known UK patient examples for reference and teaching.

- [The Role of Chest Imaging in Patient Management during the COVID-19 Pandemic](https://www.fleischner-covid19.org/)
  - Dataset description: Welcome to the Fleischner Society’s educational repository of radiographic and CT images of patients diagnosed with COVID-19. We provide this on-line supplement to our published statement as an aid to familiarize the medical community with the typical imaging findings of COVID-19. 

- [SIRM COVID-19 DATABASE](https://www.sirm.org/category/senza-categoria/covid-19/)
  - Dataset description: COVID-19 radiological cases from the Italian Society of Medical Radiology (SIRM), including images from X-rays and CT scans.

- [farmy-ai/covid-fighters](https://github.com/farmy-ai/covid-fighters)
  - Dataset description: Data collection and label tool for COVID-19 disease chest scans. 
  - Project website: [COVIDEEP](http://www.covideep.net)

- [COVID-19 Radiography Database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)
  - Dataset description: A team of researchers from Qatar University, Doha, Qatar and the University of Dhaka, Bangladesh along with their collaborators from Pakistan and Malaysia in collaboration with medical doctors have created a database of chest X-ray images for COVID-19 positive cases along with Normal and Viral Pneumonia images. 
    - Related research paper: [Can AI help in screening Viral and COVID-19 pneumonia?](https://arxiv.org/abs/2003.13145) (M.E.H. Chowdhury, T. Rahman, A. Khandakar, R. Mazhar, M.A. Kadir, Z.B. Mahbub, K.R. Islam, M.S. Khan, A. Iqbal, N. Al-Emadi, M.B.I. Reaz)
    - Code repository: [tawsifur/COVID-19-Chest-X-ray-Detection](https://github.com/tawsifur/COVID-19-Chest-X-ray-Detection)  

- [Aman9026/COVID-19-Predictor-dataset](https://github.com/Aman9026/COVID-19-Predictor-dataset)
  - Dataset description: Predict COVID-19 by analyzing chest X-Ray images in this dataset.

- [Radiopaedia](https://radiopaedia.org/)
  - Dataset description: Radiopaedia.org is a rapidly growing, open-edit radiology resource, compiled by radiologists and other health professionals from across the globe.
  - Search for COVID cases: https://radiopaedia.org/search?lang=us&page=6&q=covid&scope=cases
  - License: [Modified CC license](https://radiopaedia.org/terms)

- [Eurorad](https://www.eurorad.org/)
  - Dataset description: The purpose of Eurorad is to provide a learning environment for radiologists, radiology residents and students worldwide. It is a peer-reviewed educational tool based on radiological case reports.
  - Search for COVID cases: https://www.eurorad.org/advanced-search?search=COVID

- [SARS-COV-2 Ct-Scan Dataset](https://www.kaggle.com/plameneduardo/sarscov2-ctscan-dataset)
  - Dataset description: A large dataset of CT scans for SARS-CoV-2 (COVID-19) identification
    - Related research paper: [SARS-CoV-2 CT-scan dataset: A large dataset of real patients CT scans for SARS-CoV-2 identification](https://www.medrxiv.org/content/10.1101/2020.04.24.20078584v3) (Eduardo Soares, Plamen Angelov, Sarah Biaso, Michele Higa Froes, Daniel Kanda Abe)
    - Code repository: https://github.com/Plamen-Eduardo/xDNN-SARS-CoV-2-CT-Scan
  - License: [Creative Commons CC BY-NC-SA 4.0](https://www.kaggle.com/plameneduardo/sarscov2-ctscan-dataset)

- [China National Center for Bioinformation: 2019 Novel Coronavirus Resource (2019nCoVR)](http://ncov-ai.big.ac.cn/download?lang=en)
  - Dataset description: Dataset of the CT images and metadata are constructed from cohorts from the China Consortium of Chest CT Image Investigation (CC-CCII). All CT images are classified into novel coronavirus pneumonia (NCP) due to SARS-CoV-2 virus infection, common pneumonia and normal controls. This dataset is available globally with the aim to assist the clinicians and researchers to combat the COVID-19 pandemic.
    - Related research paper: [Clinically Applicable AI System for Accurate Diagnosis, Quantitative Measurements and Prognosis of COVID-19 Pneumonia Using Computed Tomography](https://www.cell.com/cell/fulltext/S0092-8674(20)30551-1?rss=yes) (Kang Zhang, Xiaohong Liu, Jun Shen, et al. Jianxing He, Tianxin Lin, Weimin Li, Guangyu Wang)

- [mr7495/COVID-CTset](https://github.com/mr7495/COVID-CTset)
  - Dataset description: Large Covid-19 CT scans dataset from paper: https://doi.org/10.1101/2020.06.08.20121541
    - The dataset is shared in this folder:
https://drive.google.com/drive/folders/1xdk-mCkxCDNwsMAk2SGv203rY1mrbnPB?usp=sharing
    - Related research paper: [A Fully Automated Deep Learning-based Network For Detecting COVID-19 from a New And Large Lung CT Scan Dataset](https://doi.org/10.1101/2020.06.08.20121541) (Mohammad Rahimzadeh, Abolfazl Attar, Seyed Mohammad Sakhaei)
    - Code repository: https://github.com/mr7495/COVID-CT-Code

- [aniruddh-1/COVID19_Pneumonia_detection/tree/master/ct_scan_dataset](https://github.com/aniruddh-1/COVID19_Pneumonia_detection/tree/master/ct_scan_dataset)
  - Dataset description: Images taken from https://github.com/UCSD-AI4H/COVID-CT/tree/master/Images-processed and then converted into PNG format
    - Code repository: https://github.com/aniruddh-1/COVID19_Pneumonia_detection/tree/master/codes

- [mohammad2682/Covid19-Dataset](https://github.com/mohammad2682/Covid19-Dataset)
  - Dataset description: This dataset contains 1252 CT scans that are positive for SARS-CoV-2 infection (COVID-19) and 1230 CT scans for patients non-infected by SARS-CoV-2, 2482 CT scans in total.
    - Dataset on Kaggle: http://www.kaggle.com/plameneduardo/sarscov2-ctscan-dataset
    - Related research paper: [EXPLAINABLE-BY-DESIGN APPROACH FOR COVID-19
CLASSIFICATION VIA CT-SCAN](https://eprints.lancs.ac.uk/id/eprint/143767/1/EXPLAINABLE_BY_DESIGN_APPROACH_FOR_COVID_19_CLASSI.pdf) (Plamen Angelov and Eduardo Almeida Soares)
    - Related research paper: [SARS-CoV-2 CT-scan dataset: A large dataset of real patients CT scans for SARS-CoV-2 identification](https://doi.org/10.1101/2020.04.24.20078584) (Eduardo Soares, Plamen Angelov, Sarah Biaso, Michele Higa Froes, Daniel Kanda Abe)

- [ncbi-nlp/COVID-19-CT-CXR](https://github.com/ncbi-nlp/COVID-19-CT-CXR)
  - Dataset description: COVID-19-CT-CXR is a public database of COVID-19 CXR and CT images, which are automatically extracted from COVID-19-relevant articles from the PubMed Central Open Access (PMC-OA) Subset. The annotations, relevant text, and a local copy of figures can be found at https://github.com/ncbi-nlp/COVID-19-CT-CXR/releases/
    - Related research paper: [COVID-19-CT-CXR: a freely accessible and weakly labeled chest X-ray and CT image collection on COVID-19 from biomedical literature](https://arxiv.org/abs/2006.06177) (Yifan Peng, Yu-Xing Tang, Sungwon Lee, Yingying Zhu, Ronald M. Summers, Zhiyong Lu)
    - Dataset citation: Peng Y, Tang YX, Lee S, Zhu Y, Summers RM, Lu Z. COVID-19-CT-CXR: a freely accessible and weakly labeled chest X-ray and CT image collection on COVID-19 from the biomedical literature. arxiv:2006.06177. 2020.

- [A large dataset of real patients CT scans for COVID-19 identification](https://doi.org/10.7910/DVN/SZDUQX)
  - Dataset description: We describe a publicly available multiclass CT scan dataset for SARS-CoV-2 infection identification. These data have been collected in the Public Hospital of the Government Employees of Sao Paulo (HSPM) and the Metropolitan Hospital of Lapa, both in Sao Paulo - Brazil. 
    - Related research paper: [SARS-CoV-2 CT-scan dataset: A large dataset of real patients CT scans for SARS-CoV-2 identification](https://www.medrxiv.org/content/10.1101/2020.04.24.20078584v3) (Eduardo Soares, Plamen Angelov, Sarah Biaso, Michele Higa Froes, Daniel Kanda Abe)
    - Dataset citation: Soares, Eduardo; Angelov, Plamen, 2020, "A large dataset of real patients CT scans for COVID-19 identification", https://doi.org/10.7910/DVN/SZDUQX, Harvard Dataverse, V1

- [lido1500/Extensive-and-Augmented-COVID-19-X-Ray-and-CT-Chest-Images-Dataset-](https://github.com/lido1500/Extensive-and-Augmented-COVID-19-X-Ray-and-CT-Chest-Images-Dataset-)
  - Dataset description: This COVID-19 dataset consists of Non-COVID and COVID cases of both X-ray and CT images. The associated dataset is augmented with different augmentation techniques to generate about 17100 X-ray and CT images. 
    - You can download this dataset from the following link: https://data.mendeley.com/datasets/8h65ywd2jr/2
    - Dataset citation: El-Shafai, Walid; E. Abd El-Samie, Fathi (2020), “Extensive and Augmented COVID-19 X-Ray and CT Chest Images Dataset”, Mendeley Data, v2
    - Other dataset versions: [v1](https://data.mendeley.com/datasets/8h65ywd2jr/1), [v2](https://data.mendeley.com/datasets/8h65ywd2jr/2), [v3](https://data.mendeley.com/datasets/8h65ywd2jr/3)

- [BrixIA: COVID19 severity score assessment project and database](https://brixia.github.io/)
  - Dataset description: We designed an end-to-end deep learning architecture for predicting, on Chest X-rays images (CRX), a multi-regional score conveying the degree of lung compromise in COVID-19 patients. Such scoring system, namely Brixia score, was applied in serial monitoring of such patients, showing significant prognostic value, in one of the hospitals that experienced one of the highest pandemic peaks in Italy. Moreover, we provide the full dataset with the related metadata and Brixia score annotations, and the code to reproduce our results.
    - Related research paper: [End-to-end learning for semiquantitative rating of COVID-19 severity on Chest X-rays](https://arxiv.org/abs/2006.04603) (Alberto Signoroni, Mattia Savardi, Sergio Benini, Nicola Adami, Riccardo Leonardi, Paolo Gibellini, Filippo Vaccher, Marco Ravanelli, Andrea Borghesi, Roberto Maroldi, Davide Farina (University of Brescia))
    - Code repository: [BrixIA/Brixia-score-COVID-19](https://github.com/BrixIA/Brixia-score-COVID-19)

- [v7labs/covid-19-xray-dataset](https://github.com/v7labs/covid-19-xray-dataset)
  - Dataset description: 12000+ manually drawn pixel-level lung segmentations, with and without covid. **WARNING:** This dataset is not intended for use in clinical diagnostics.
    - Browse & download the dataset on V7 Darwin here: https://darwin.v7labs.com/v7-labs/covid-19-chest-x-ray-dataset

- [GeneralBlockchain/covid-19-chest-xray-segmentations-dataset](https://github.com/GeneralBlockchain/covid-19-chest-xray-segmentations-dataset)
  - Dataset description: Segmentations of COVID-19 Chest X-ray Dataset.

- [GeneralBlockchain/covid-19-chest-xray-lung-bounding-boxes-dataset](https://github.com/GeneralBlockchain/covid-19-chest-xray-lung-bounding-boxes-dataset)
  - Dataset description: Lung Bounding Boxes of COVID-19 Chest X-ray Dataset.





## Software Projects and Tutorials

- [JordanMicahBennett/SMART-CT-SCAN_BASED-COVID19_VIRUS_DETECTOR](https://github.com/JordanMicahBennett/SMART-CT-SCAN_BASED-COVID19_VIRUS_DETECTOR)
  - Project description: An image based Xray attempt at coronavirus2019 (covid19) diagnosis using a convolutional neural network design.
  - License: [MIT License](https://github.com/JordanMicahBennett/SMART-CT-SCAN_BASED-COVID19_VIRUS_DETECTOR/blob/master/LICENSE.txt)

- [haydengunraj/COVIDNet-CT](https://github.com/haydengunraj/COVIDNet-CT)
  - Project description: COVID-Net Open Source Initiative - Models for COVID-19 Detection from Chest CT
  - License: [GNU Affero General Public License (AGPL) v3.0](https://github.com/haydengunraj/COVIDNet-CT/blob/master/LICENSE.md)

- [lindawangg/COVID-Net](https://github.com/lindawangg/COVID-Net)
  - Project description: COVID-Net Open Source Initiative
    - Related research paper: [COVID-Net: A Tailored Deep Convolutional Neural Network Design for Detection of COVID-19 Cases from Chest Radiography Images](https://arxiv.org/abs/2003.09871) (Linda Wang, Zhong Qiu Lin, and Alexander Wong; Department of Systems Design Engineering, University of Waterloo, Canada; Waterloo Artificial Intelligence Institute, Canada; DarwinAI Corp., Canada)
    - Related dataset: [agchung/Figure1-COVID-chestxray-dataset](https://github.com/agchung/Figure1-COVID-chestxray-dataset)
  - License: [GNU Affero General Public License (AGPL) v3.0](https://github.com/lindawangg/COVID-Net/blob/master/LICENSE.md)

- [IliasPap/COVIDNet](https://github.com/IliasPap/COVIDNet)
  - Project description: PyTorch implementation of [COVID-Net](https://github.com/lindawangg/COVID-Net)
  - License: [GNU General Public License (GPL) v3.0](https://github.com/IliasPap/COVIDNet/blob/master/LICENSE)

- [zeeshannisar/COVID-19](https://github.com/zeeshannisar/COVID-19)
  - Project description: Detecting and Visualising the Infectious Regions of COVID-19 in X-ray Images Using Different Pretrained-Networks in Tensorflow 2.x.

- [Chester the AI Radiology Assistant](https://github.com/mlmed/dl-web-xray)
  - Project description: NOT FOR MEDICAL USE. This is a prototype system for diagnosing chest x-rays using neural networks.
  - Implementations: 
    - In-browser web app: https://mlmed.org/tools/xray/
    - Downloadable macOS app: https://github.com/mlmed/dl-web-xray/releases/download/2.0/Chester.app.zip

- [sydney0zq/covid-19-detection](https://github.com/sydney0zq/covid-19-detection)
  - Project description: The implementation of "Deep Learning-based Detection for COVID-19 from Chest CT using Weak Label".
    - Related research paper: [Deep Learning-based Detection for COVID-19 from Chest CT using Weak Label](https://www.medrxiv.org/content/10.1101/2020.03.12.20027185v2) (Chuangsheng Zheng, Xianbo Deng, Qiang Fu, Qiang Zhou, Jiapei Feng, Hui Ma, Wenyu Liu, Xinggang Wang;
Department of Radiology, Union Hospital, Tongji Medical College, Huazhong University of Science and Technology, Wuhan, 430022, China; Hubei Province Key Laboratory of Molecular Imaging, Wuhan, 430022, China; Artificial Intelligence Institute, School of Electronic Information and Communications, Huazhong University of Science and Technology, Wuhan, 430074, China) (Preprints: [v1](https://www.medrxiv.org/content/10.1101/2020.03.12.20027185v1.full.pdf), [v2](https://www.medrxiv.org/content/10.1101/2020.03.12.20027185v2))
  - Online implementation: http://39.100.61.27/ This website provides online testing on user-provided CT volume, and the results are the probabilities of being a normal person and being infected by COVID-19.
  - License: [Creative Commons CC-BY-NC-SA-4.0](https://github.com/sydney0zq/covid-19-detection/blob/master/CC-BY-NC-SA-4.0)
  - Related GitHub Gist: [convert_dicom_to_npy.py](https://gist.github.com/sydney0zq/4813618fd92781618e3c90809fc1db8b)

- [elcronos/COVID-19](https://github.com/elcronos/COVID-19) (Predecessor of [FightCOVID19](https://github.com/FightCOVID19))
  - Project description: COVID-19 Detector from x-rays using Computer Vision and Deep Learning
  - License: COVID-19 Detector by Camilo Pestana is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/).

- [FightCOVID19](https://github.com/FightCOVID19)
  - Project description: Fight COVID-19 is a non-profit, collaborative community democratising AI to assist in the detection and triage of COVID-19 cases
  - Project website (with information on upcoming Deepvision app): https://www.fightcovid19.ai/

- [PyTorchLightning/lightning-Covid19](https://github.com/PyTorchLightning/lightning-Covid19)
  - Project description: A detector for covid-19 chest X-ray images using PyTorch Lightning (for educational purposes)

- [Detecting COVID-19 in X-ray images with Keras, TensorFlow, and Deep Learning](https://www.pyimagesearch.com/2020/03/16/detecting-covid-19-in-x-ray-images-with-keras-tensorflow-and-deep-learning/)
  - Project description: In this tutorial, you will learn how to automatically detect COVID-19 in a hand-created X-ray image dataset using Keras, TensorFlow, and Deep Learning.
  - Availability: (Source code can be downloaded upon request via email)

- [AleGiovanardi/covidhelper](https://github.com/AleGiovanardi/covidhelper)
  - Project description: Detect COVID19 infection from RX and CT scans
  

- [rekalantar/covid19_detector](https://github.com/rekalantar/covid19_detector)
  - Project description: Using Deep Learning to detect Covid-19 from X-Ray/CT scans of patients

- [bkong999/COVNet](https://github.com/bkong999/COVNet)
  - Project description: This is a PyTorch implementation of the paper "[Artificial Intelligence Distinguishes COVID-19 from Community Acquired Pneumonia on Chest CT](https://pubs.rsna.org/doi/10.1148/radiol.2020200905)". It supports training, validation and testing for COVNet.

- [ahmed3991/Covid-19-X-Rays-Detector](https://github.com/ahmed3991/Covid-19-X-Rays-Detector)
  - Project description: Detect Covid-19 infection from X-Rays

- [chiragsamal/COVID19-Detection](https://github.com/chiragsamal/COVID19-Detection)
  - Project description: CoronaVirus (COVID-19) detection using X_Ray Images

- [Synthetic.Network](https://synthetic.network/)
  - Project description: NOT FOR MEDICAL USE. This is a prototype of a deep learning tool to diagnose frontal chest X-ray images and recognize bacterial pneumonia, viral pneumonia and coronavirus. 

- [aildnont/covid-cxr](https://github.com/aildnont/covid-cxr)
  - Project description: Neural network model for classifying chest X-rays by presence of COVID-19 features
  - License: [MIT License](https://github.com/aildnont/covid-cxr/blob/master/LICENSE)

- [velebit-ai/COVID-Next-Pytorch](https://github.com/velebit-ai/COVID-Next-Pytorch)
  - Project description: COVID-Next -> Pytorch upgrade of the COVID-Net for COVID-19 detection in X-Ray images
  - License: [MIT License](https://github.com/velebit-ai/COVID-Next-Pytorch/blob/master/LICENSE)

- [manujosephv/covid-xray-imagenet](https://github.com/manujosephv/covid-xray-imagenet)
  - Project description: Imagenet Pretraining for Covid-19 Xray Identification
    - Related blog post: [Does Imagenet Pretraining work for Chest Radiography Images(COVID-19)?](https://deep-and-shallow.com/2020/04/05/does-imagenet-pretraining-work-for-chest-radiography-imagescovid-19/)
  - License: [MIT License](https://github.com/manujosephv/covid-xray-imagenet/blob/master/LICENSE)

- [sagarnildass/covid_19_xray_classification](https://github.com/sagarnildass/covid_19_xray_classification)
  - Project description: In this repo, we will explore the methodology of implementing AI Based solutions for comparing COVID-19 against other types of viral illness, and will analyze the results obtained.

- [adeaeede/ai4covid](https://github.com/adeaeede/ai4covid)
  - Project description: This project is a result of the [WirVsVirus Hackathon](https://wirvsvirushackathon.org/). Our objective was to assist doctors in diagnosing COVID-19 patients by training a convolutional network to discriminate between patients with and without the disease, based on an X-ray image.

- [ChirilaLaura/COVID-Z](https://github.com/ChirilaLaura/COVID-Z)
  - Project description: An online AI based platform for COVID-19 testing.
  - Web page on Devpost: https://devpost.com/software/covid-z
  - License: [MIT License](https://github.com/ChirilaLaura/COVID-Z/blob/master/LICENSE)

- [ChirilaLaura/COVID-X](https://github.com/ChirilaLaura/COVID-X) ([V1](https://github.com/ChirilaLaura/COVID-X), [V2](https://github.com/ChirilaLaura/COVID-X_V2), [V3](https://github.com/ChirilaLaura/COVID-X_V3))
  - Project description: *(No description, website, or topics provided)*
  - License: [MIT License](https://github.com/ChirilaLaura/COVID-X/blob/master/LICENSE)

- [hananshafi/covid19-detection](https://github.com/hananshafi/covid19-detection)
  - Project description: This code is for predicting COVID-19 from chest Xrays.

- [tawsifur/COVID-19-Chest-X-ray-Detection](https://github.com/tawsifur/COVID-19-Chest-X-ray-Detection)
  - Project description: A team of researchers from Qatar University, Doha, Qatar and the University of Dhaka, Bangladesh along with their collaborators from Pakistan and Malaysia in collaboration with medical doctors have managed to classify COVID-19, Viral pneumonia and Normal Chest X-ray images with an accuracy of 98.3%.
    - Related research paper: [Can AI help in screening Viral and COVID-19 pneumonia?](https://arxiv.org/abs/2003.13145) (M.E.H. Chowdhury, T. Rahman, A. Khandakar, R. Mazhar, M.A. Kadir, Z.B. Mahbub, K.R. Islam, M.S. Khan, A. Iqbal, N. Al-Emadi, M.B.I. Reaz)
    - Related dataset: [COVID-19 Radiography Database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)

- [talhaanwarch/Corona_Virus](https://github.com/talhaanwarch/Corona_Virus)
  - Project description: Diagnosis of corona virus using Chest Xray, through deep learning

- [defeatcovid19/defeatcovid19-net-pytorch](https://github.com/defeatcovid19/defeatcovid19-net-pytorch)
  - Project description: Pytorch solution for predictions on X-ray images of COVID-19 patients
  - License: [MIT License](https://github.com/defeatcovid19/defeatcovid19-net-pytorch/blob/master/LICENSE)

- [itratrahman/covid_19](https://github.com/itratrahman/covid_19)
  - Project description: Project to detect COVID19 from X-Rays.

- [saisriteja/covid19](https://github.com/saisriteja/covid19)
  - Project description: COVID image data analysis using chest X-rays and CT scans.

- [BioXAI/DeepCOVIDExplainer](https://github.com/BioXAI/DeepCOVIDExplainer)
  - Project description: Explainable COVID-19 Diagnosis from Chest Radiography Images

- [appushona123/Instant-COVID19-detection](https://github.com/appushona123/Instant-COVID19-detection)
  - Project description: This project is able to detect the COVID19 patient with the X-ray image of the person. 

- [talhaanwarch/Corona_Virus](https://github.com/talhaanwarch/Corona_Virus)
  - Project description: Diagnosis of corona virus using Chest Xray, through deep learning

- [karanshah7371/Covid19Xray](https://github.com/karanshah7371/Covid19Xray)
  - Project description: This model predicts covid-19 (if we feed image of X-Ray of patients). DISCLAIMER: This model is currently in development and we do not recommend using it for screening purposes as of now.
  - Online implementation: http://www.covidfilter.life/
  - License: [GNU General Public License (GPL) v3.0](https://github.com/karanshah7371/Covid19Xray/blob/master/LICENSE)

- [devcode925/covid19XraySample](https://github.com/devcode925/covid19XraySample)
  - Project description: COVID-19 Xray Study using CNN from deeplearning4J library.

- [arunslb123/covid19_xrays](https://github.com/arunslb123/covid19_xrays)
  - Project description: Covid X-ray prediction model
  - Online implementation: http://covid.mlproducts.app/

- [xinli0928/COVID-Xray](https://github.com/xinli0928/COVID-Xray)
  - Project description: We present COVID-MobileXpert: a lightweight deep neural network (DNN) based mobile app that can use noisy snapshots of chest X-ray (CXR) for point-of-care COVID-19 screening. 
    - Related research paper: "[COVID-MobileXpert: On-Device COVID-19 Screening using Snapshots of Chest X-Ray](https://arxiv.org/pdf/2004.03042.pdf)" (Xin Li, Chengyin Li, Dongxiao Zhu).
  - Android app: https://drive.google.com/file/d/1yqNsVHkrrCoo_XYedOOqSUJRnzc0vjIB/view?usp=sharing
  - License: [MIT License](https://github.com/xinli0928/COVID-Xray/blob/master/LICENSE)

- [linhduongtuan/Covid-19_Xray_Classifier](https://github.com/linhduongtuan/Covid-19_Xray_Classifier)
  - Project description: Demo diagnosis tools for Covid-19 Chest Xray
    - Related research paper: [Deep Learning for Automated Recognition of Covid-19 from Chest X-ray Images](https://www.medrxiv.org/content/10.1101/2020.08.13.20173997v1) (Phuong Nguyen, Ludovico Iovino, Michele Flammini, Linh Tuan Linh)
  - License: [MIT License](https://github.com/linhduongtuan/Covid-19_Xray_Classifier/blob/master/LICENSE)

- [ufopcsilab/EfficientNet-C19](https://github.com/ufopcsilab/EfficientNet-C19)
  - Project description: Repository to reproduce the results of "Towards an Effective and Efficient Deep Learning Model for COVID-19 Patterns Detection in X-ray Images"
    - Related research paper: [Towards an Effective and Efficient Deep Learning Model for COVID-19 Patterns Detection in X-ray Images](https://arxiv.org/abs/2004.05717) (Eduardo Luz, Pedro Lopes Silva, Rodrigo Silva, Ludmila Silva, Gladston Moreira, David Menotti)
  - License: [GNU Affero General Public License (AGPL) v3.0](https://github.com/ufopcsilab/EfficientNet-C19/blob/master/LICENSE)

- [ufopcsilab/covid-19](https://github.com/ufopcsilab/covid-19)
  - Project description: Based on the work "Towards an Effective and Efficient Deep Learning Model for COVID-19 Patterns Detection in X-ray Images"
    - Related research paper: [Towards an Effective and Efficient Deep Learning Model for COVID-19 Patterns Detection in X-ray Images](https://arxiv.org/abs/2004.05717) (Eduardo Luz, Pedro Lopes Silva, Rodrigo Silva, Ludmila Silva, Gladston Moreira, David Menotti)
  - Project web page: [iCOVID-X: COVID-19 Detection in X-Ray Images using AI](http://www.decom.ufop.br/csilab/projects/)
  - License: [MIT License](https://github.com/ufopcsilab/covid-19/blob/master/LICENSE)

- [kaushikjadhav01/COVID-19-Detection-Flask-App-based-on-Chest-X-rays-and-CT-Scans](https://github.com/kaushikjadhav01/COVID-19-Detection-Flask-App-based-on-Chest-X-rays-and-CT-Scans)
  - Project description: COVID-19 Detection based on Chest X-rays and CT Scans using four Transfer Learning algorithms: VGG16, ResNet50, InceptionV3, Xception.
    - Related blog post: [COVID-19 Detector Flask App based on Chest X-rays and CT Scans using Deep Learning](https://towardsdatascience.com/covid-19-detector-flask-app-based-on-chest-x-rays-and-ct-scans-using-deep-learning-a0db89e1ed2a)

- [amobiny/DECAPS_for_COVID19](https://github.com/amobiny/DECAPS_for_COVID19)
  - Project description: Official PyTorch implementation of the Detail-oriented Capsule Network (DECAPS) proposed in the paper Radiologist-Level COVID-19 Detection Using CT Scans with Detail-Oriented Capsule Networks.
    - Related research paper: [Radiologist-Level COVID-19 Detection Using CT Scans with Detail-Oriented Capsule Networks](https://arxiv.org/pdf/2004.07407.pdf) (Aryan Mobiny, Pietro A. Cicalese, Samira Zare, Pengyu Yuan, Mohammad S. Abavisan, Carol C. Wu, Jitesh Ahuja, Patricia M. de Groot, and Hien V. Nguyen)
  - License: [MIT License](https://github.com/amobiny/DECAPS_for_COVID19/blob/master/LICENSE)

- [junqiangchen/COVID-CT19-Challenge](https://github.com/junqiangchen/COVID-CT19-Challenge)
  - Project description: This is an example of classifying each CT image into positive COVID-19 (the image has clinical findings of COVID-19) or negative COVID-19 ( the image does not have clinical findings of COVID-19).

- [mr7495/COVID-CT-Code](https://github.com/mr7495/COVID-CT-Code)
  - Project description: Fully automated code for Covid-19 detection from CT scans from paper: https://doi.org/10.1101/2020.06.08.20121541
    - Related research paper: [A Fully Automated Deep Learning-based Network For Detecting COVID-19 from a New And Large Lung CT Scan Dataset](https://doi.org/10.1101/2020.06.08.20121541) (Mohammad Rahimzadeh, Abolfazl Attar, Seyed Mohammad Sakhaei)
    - Related dataset: https://github.com/mr7495/COVID-CTset

- [mr7495/covid19](https://github.com/mr7495/covid19)
  - Project description: Covid-19 and Pneumonia detection from X-ray Images
    - Related research paper: [A modified deep convolutional neural network for detecting COVID-19 and pneumonia from chest X-ray images based on the concatenation of Xception and ResNet50V2](https://doi.org/10.1016/j.imu.2020.100360) (Mohammad Rahimzadeh, Abolfazl Attar)

- [AshleyRudland/covid19](https://github.com/AshleyRudland/covid19)
  - Project description: Upload images of CT/xray scans to see if a patient has COVID19. Not for production use yet, first POC.
    - Related blog post: [Detecting COVID-19 from X-Ray/CT Scans using AI with 98% Accuracy (Deep Learning) - Lesson #1](https://ashleyrudland.com/2020/03/16/detecting-covid19-with-ai.html)

- [paulinawins/CovidProject](https://github.com/paulinawins/CovidProject)
  - Project description: Covid CT Scan Detection Web Application

- [aniruddh-1/COVID19_Pneumonia_detection/tree/master/codes](https://github.com/aniruddh-1/COVID19_Pneumonia_detection/tree/master/codes)
  - Project description: Detects Covid-19 Pneumonia signs from CT Scan Images by a CNN Model
    - Related dataset: https://github.com/aniruddh-1/COVID19_Pneumonia_detection/tree/master/ct_scan_dataset

- [MI-12/BigBIGAN-for-COVID-19](https://github.com/MI-12/BigBIGAN-for-COVID-19)
  - Project description: Code and resorces of the study "End-to-end automatic detection of the Coronavirus disease 2019 (COVID-19) based on CT image"

- [SUMEETRM/covid19-ai](https://github.com/SUMEETRM/covid19-ai)
  - Project description: The following model aims to present a neural network aimed to detect COVID-19 cases through chest X-Rays.
  - License: [GNU General Public License (GPL) v3.0](https://github.com/SUMEETRM/covid19-ai/blob/master/LICENSE)

- [Coronavirus-Visualization-Team/CVT_COVID-CT-CNN](https://github.com/Coronavirus-Visualization-Team/CVT_COVID-CT-CNN)
  - Project description: Collaboration space/data storage for development of a COVID CT CNN by CVT

- [includeamin/COVID-19](https://github.com/includeamin/COVID-19)
  - Project description: Detection of covid-19 from X-ray images Using keras and tensorflow.
  - License: [Apache License 2.0](https://github.com/includeamin/COVID-19/blob/master/LICENSE)

- [hortonworks-sk/CML-COVID-CT](https://github.com/hortonworks-sk/CML-COVID-CT)
  - Project description: Diagnosing COVID-19 - Classification of Ground Glass Opacities in CT scans using CML
    - Related research paper: [Correlation of Chest CT and RT-PCR Testing for Coronavirus Disease 2019 (COVID-19) in China: A Report of 1014 Cases](https://doi.org/10.1148/radiol.2020200642) (Tao Ai, Zhenlu Yang, Hongyan Hou, Chenao Zhan, Chong Chen, Wenzhi Lv, Qian Tao, Ziyong Sun, Liming Xia)

- [ThaisLuca/WNN-Covid-CT](https://github.com/ThaisLuca/WNN-Covid-CT)
  - Project description: Application of Weightless Neural Networks (WNNs) for Covid-19 detection in CT images. 

- [aaron2181/CovidProject](https://github.com/aaron2181/CovidProject)
  - Project description: Covid CT Scan Detection Web Application

- [Prerna5194/COVID-19-CT-Classification](https://github.com/Prerna5194/COVID-19-CT-Classification)
  - Project description: COVID 19 CT Image Classification

- [strcoder4007/COVID-19-Deep-Learning](https://github.com/strcoder4007/COVID-19-Deep-Learning)
  - Project description: Detecting COVID-19 using X-ray Images, CT Scans and Deep Learning

- [shrey-bansal/COVID19-CTSCAN](https://github.com/shrey-bansal/COVID19-CTSCAN)
  - Project description: COVID 19 CT SCAN Screening Model
  - Online implementation: https://covid19-ctscan.herokuapp.com/

- [josehernandezsc/COVID19Net](https://github.com/josehernandezsc/COVID19Net)
  - Project description: COVID19 CT Scan Visual Recognition Project

- [apascuet/covID](https://github.com/apascuet/covID)
  - Project description: Use of different convolutional neural networks for classification of COVID-19 2D CT images.

- [reva2498/Medical-Imaging-Samhar-COVID19](https://github.com/reva2498/Medical-Imaging-Samhar-COVID19)
  - Project description: Detecting COVID-19 in CT Scan images

- [devkumar07/COVID19-CTScanClassifier](https://github.com/devkumar07/COVID19-CTScanClassifier)
  - Project description: This deep learning model is trained using the CNN model to determine whether the patient is normal or has pneumonia based on the patient's CT Scan image. Furthermore, it can classify the virus that is causing the respiratory illness including COVID-19 with overall accuracy of >90%

- [shawon100/Covid-19-Disease-Diagnosis](https://github.com/shawon100/Covid-19-Disease-Diagnosis)
  - Project description: Covid-19 Diagnosis Python Flask Web App . It can detect COVID-19 from CT Scan Medical Images

- [tkseneee/COVID19_Image_Classification](https://github.com/tkseneee/COVID19_Image_Classification)
  - Project description: Basic CNN model for classifying COVID19 from CT chest image.

- [shengrenhou/COVID-19-Detection](https://github.com/shengrenhou/COVID-19-Detection)
  - Project description: Detecting COVID-19 by Resnet neural network
  - License: [MIT License](https://github.com/shengrenhou/COVID-19-Detection/blob/master/LICENSE)

- [jiangdat/COVID-19](https://github.com/jiangdat/COVID-19)
  - Project description: Deep Learning for COVID-19 chest CT image analysis

- [am-uff/covid19-ct](https://github.com/am-uff/covid19-ct)
  - Project description: Diagnosis of Covid-19 based on chest CT using Deep Learning (Diagnóstico da Covid-19 com base em TC de Tórax utilizando Deep Learning)

- [raunak-sood2003/COVID-19-Classification](https://github.com/raunak-sood2003/COVID-19-Classification)
  - Project description: Automated Classification of COVID-19 in lung CT scans
    - Related article: [Using pretrained deep convolutional neural networks for binary classification of COVID-19 CT scans](https://towardsdatascience.com/using-pretrained-deep-convolutional-neural-networks-for-binary-classification-of-covid-19-ct-scans-3a7f7ea8b543)

- [ThomasMerkh/covid-ct](https://github.com/ThomasMerkh/covid-ct)
  - Project description: This repository contains a deep convolutional network trained on CT data for the binary classification of COVID/Non-COVID. Transfer learning was used here, where I utilized a pre-trained COVID-Net model (see https://arxiv.org/abs/2003.09871v1), and fine-tuned the parameters of the network using the training set. 

- [bigvisionai/covid-ct-challenge](https://github.com/bigvisionai/covid-ct-challenge)
  - Project description: This repo contains the code for the COVID-CT Challenge ("CT diagnosis of COVID-19," https://covid-ct.grand-challenge.org/CT-diagnosis-of-COVID-19)
  - License: [MIT License](https://github.com/bigvisionai/covid-ct-challenge/blob/master/LICENSE)

- [shuxg2017/COVID19-CTscan-image-classification](https://github.com/shuxg2017/COVID19-CTscan-image-classification)
  - Project description: In this project, we were trying to classify NonCOVID and COVID CT scan images. We built an autoencoder to extract latent space features. Then by using the latent space features, we can classify the images by using K nearest neighbor and Bayesian model.

- [Divyakathirvel26/Covid-19](https://github.com/Divyakathirvel26/Covid-19)
  - Project description: Detection of Covid-19 using CNN with X-ray and CT-scan images

- [alext234/covid19-models-testing](https://github.com/alext234/covid19-models-testing)
  - Project description: Testing various deep learning models that detect COVID-19 based on X-ray images or CT scan images
  - License: [Apache License 2.0](https://github.com/alext234/covid19-models-testing/blob/master/LICENSE)

- [T-SHARMILA/Covid-19-CT-scans](https://github.com/T-SHARMILA/Covid-19-CT-scans)
  - Project description: Classification of Covid-19 CT scans and Non-covid CT scans using Transfer Learning (VGG-16 model)

- [bjmcshane/Covid-19_CTscans](https://github.com/bjmcshane/Covid-19_CTscans)
  - Project description: My convolutional neural network built to diagnose coronavirus by looking at CT scans.

- [varna30/COVID-19-Classifier](https://github.com/varna30/COVID-19-Classifier)
  - Project description: Detection of COVID-19 from standard Machine Learning Algorithms using CT scan images dataset

- [Ayazdi/COVID-19-Diagnosis](https://github.com/Ayazdi/COVID-19-Diagnosis)
  - Project description: A web based engine to diagnose COVID-19 by CT scans
  - License: [MIT License](https://github.com/Ayazdi/COVID-19-Diagnosis/blob/master/LICENSE)

- [jlcadavid/COVID-19_CNN_Project](https://github.com/jlcadavid/COVID-19_CNN_Project)
  - Project description: COVID-19 (Coronavirus) CT Scanner using Convolutional Neural Network (CNN) classifier.

- [pourabkarchaudhuri/covid-19-classification-x-ray](https://github.com/pourabkarchaudhuri/covid-19-classification-x-ray)
  - Project description: Deep Learning based COVID-19 affect lungs CT-scan classification to healthy or infected

- [aayush9400/Covid-19-CT-SCAN-Classifier](https://github.com/aayush9400/Covid-19-CT-SCAN-Classifier)
  - Project description: This is a project with a working website integrated with a CNN model to make predictions whether a patient is covid-19 positive or not

- [yaseenaiman/COVID19_detector](https://github.com/yaseenaiman/COVID19_detector)
  - Project description: A deep learning algorithm using Colab and Tensorflow to predict whether or not a person has COVID-19 by uploading his/her CT scan
    - Related blog post: https://yaseenaimanmohammed.wordpress.com/2020/04/06/covid-19-detector/

- [moklesur1993/COVID-Attention](https://github.com/moklesur1993/COVID-Attention)
  - Project description: An Attention based deep neural network for recognition of COVID-19 cases from CT images

- [xiaoxuegao499/LA-DNN-for-COVID-19-diagnosis](https://github.com/xiaoxuegao499/LA-DNN-for-COVID-19-diagnosis)
  - Project description: Online COVID-19 diagnosis with chest CT images: Lesion-attention deep neural networks

- [shujams/CT-AI-Deep-Learning-Project](https://github.com/shujams/CT-AI-Deep-Learning-Project)
  - Project description: 2019-nCoV Categorization and Prediction: A Deep Learning Analysis. Thoracic CT Scan Classification and Probable Diagnosis of COVID-19 Utilizing ML/Fast.ai Libraries.

- [arthursdays/HKBU_HPML_COVID-19](https://github.com/arthursdays/HKBU_HPML_COVID-19)
  - Project description: Source code of paper "Benchmarking Deep Learning Models and Automated Model Design for COVID-19 Detection with Chest CT Scans".
    - Related research paper: [Benchmarking Deep Learning Models and Automated Model Design for COVID-19 Detection with Chest CT Scans](https://www.medrxiv.org/content/10.1101/2020.06.08.20125963v2) (Xin He, Shihao Wang, Shaohuai Shi, Xiaowen Chu, Jiangping Tang, Xin Liu, Chenggang Yan, Jiyong Zhang, Guiguang Ding)

- [saanville/CovidCT](https://github.com/saanville/CovidCT)
  - Project description: Detection of COVID-19 by classifying Lungs CT scan

- [ilmimris/ct-covid19-model](https://github.com/ilmimris/ct-covid19-model)
  - Project description: Detecting COVID-19 in X-ray images with Keras, TensorFlow, and Deep Learning

- [GastonLagaffe2013/Covid-19-Classifier](https://github.com/GastonLagaffe2013/Covid-19-Classifier)
  - Project description: Classify COVID-19 CT images using ResNeXt and other networks

- [amartinez2020/COVID_CT](https://github.com/amartinez2020/COVID_CT)
  - Project description: Classification of COVID-19 in CT Scans using Multi-Source Transfer Learning

- [Arindam-coder/COVID19-Detection-from-Chest-CT-Scan](https://github.com/Arindam-coder/COVID19-Detection-from-Chest-CT-Scan)
  - Project description: Image classification project using CNN. I have used a dataset which I have got from Harvard University’s site, https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FSZDUQX
  - License: [MIT License](https://github.com/Arindam-coder/COVID19-Detection-from-Chest-CT-Scan/blob/master/LICENSE)

- [junyuchen245/COVID19_CT_Segmentation_3DSlicer](https://github.com/junyuchen245/COVID19_CT_Segmentation_3DSlicer)
  - Project description: An extension module in [3DSlicer](https://www.slicer.org/) for COVID19 CT segmentation using Convolutional Neural Networks.
  - License: [MIT License](https://github.com/junyuchen245/COVID19_CT_Segmentation_3DSlicer/blob/master/LICENSE)

- [themendu/covid_from_ctscan](https://github.com/themendu/covid_from_ctscan)
  - Project description: Model which can predict COVID-19 positive case from axial lung CT-scan images.

- [shervinmin/DeepCovid](https://github.com/shervinmin/DeepCovid)
  - Project description: In this repository, we provide the PyTorch implementation of the DeepCovid Framework (the training and inference code) for the research community to use.
    - Related research paper: [Deep-COVID: Predicting COVID-19 From Chest X-Ray Images Using Deep Transfer
Learning](https://arxiv.org/pdf/2004.09363.pdf) (Shervin Minaeea, Rahele Kafiehb, Milan Sonkac, Shakib Yazdanid, Ghazaleh Jamalipour Souf)
    - Related dataset: [COVID-XRay-5K DATASET](https://github.com/shervinmin/DeepCovid/tree/master/data) from two sources: [Covid-Chestxray-Dataset](https://github.com/ieee8023/covid-chestxray-dataset) for COVID-19 X-ray samples, and [ChexPert Dataset](https://stanfordmlgroup.github.io/competitions/chexpert/) for Non-COVID samples.

- [arunpandian7/covid-19-detection](https://github.com/arunpandian7/covid-19-detection)
  - Project description: Covid-19 Detection using CT Scan of Lung Infections with Deep Learning and Computer Vision
  - Flask app for deployment: https://github.com/arunpandian7/covid-detector-flask
    - Online demo: https://covid-xray-detector.herokuapp.com/

- [TianxingWang0/COVID-19-CT-Diagnose](https://github.com/TianxingWang0/COVID-19-CT-Diagnose)
  - Project description: Pretrained (imagenet) ResNet50 with 3 classification.

- [sensynehealth/lung-ct-covid-pathology-detection](https://github.com/sensynehealth/lung-ct-covid-pathology-detection)
  - Project description: Early exploration of detecting signs of lung pathology in COVID-19 Lung CT cases
    - Related blog post:  "A weekend, a ‘virtual’ hackathon and ML approaches to automate the analysis of COVID-19 lung CT scans" ([Sensyne Health](https://www.sensynehealth.com/newsroom/a-weekend-a-virtual-hackathon-and-ml-approaches-to-automate-the-analysis-of-covid-19-lung-ct-scans), [Medium](https://medium.com/@heaven_spring_cheetah_212/a-weekend-a-virtual-hackathon-and-ml-approaches-to-automate-the-analysis-of-covid-19-lung-ct-5ad4c89fe03c))
  - License: [MIT License](https://github.com/sensynehealth/lung-ct-covid-pathology-detection/blob/master/LICENSE)

- [FarheenB/covid-detection-ct-scan-images](https://github.com/FarheenB/covid-detection-ct-scan-images)
  - Project description: Detect whether a person is COVID-19 positive by CT Scan images of Transverse Section of Chest.

- [andi-iqbal/covid-chestxray](https://github.com/andi-iqbal/covid-chestxray)
  - Project description: A Machine Learning project with aim to predict COVID-19 diagnosis using X-Ray and CT Scan Images

- [Firoxon/covid-flask-deploy](https://github.com/Firoxon/covid-flask-deploy)
  - Project description: The deploy-able solution of ML Based COVID Detector using X-ray and CT Scans of Lungs
  - License: [MIT License](https://github.com/Firoxon/covid-flask-deploy/blob/master/LICENSE)

- [AdityaPotluri/CoronaCT](https://github.com/AdityaPotluri/CoronaCT)
  - Project description: Determines whether a given CT scan has coronavirus or not.

- [melanieshi0120/COVID_19_chest_CT_Image_Classification](https://github.com/melanieshi0120/COVID_19_chest_CT_Image_Classification)
  - Project description: Image Classification: Neural Network/ Random Forest

- [durvesh8/COVID-19-Classifier-based-on-CT-Scans](https://github.com/durvesh8/COVID-19-Classifier-based-on-CT-Scans)
  - Project description: A classifier based on Tensorflow and Keras. The model which was used for this classifier was DenseNet121. 

- [sandijean90/covid-19](https://github.com/sandijean90/covid-19)
  - Project description: Using FastAI/PyTorch to create a CNN (Convolutional Neural Network) to predict whether an X-ray or CT Image is COVID-19 positive or negative (binary output).

- [nnassime/A-simple-and-useful-code-for-covid-19-detection-on-CT-Scans](https://github.com/nnassime/A-simple-and-useful-code-for-covid-19-detection-on-CT-Scans)
  - Project description: A simple tensorflow and keras code useful for the covid 19 detection on CT scans.

- [Rushikesh042/COVID-19-Detection-using-Deep-Learning](https://github.com/Rushikesh042/COVID-19-Detection-using-Deep-Learning)
  - Project description: Development of a deep learning-based model for automatic Covid-19 detection on chest CT to counter the outbreak of SARS-CoV-2

- [umarshakeb/WWCode-Image_Classification](https://github.com/umarshakeb/WWCode-Image_Classification)
  - Project description: The project aims to use Chest X-Ray and CT Scan images to identify patients who are COVID positive.

- [modabbir24/Build-COVID19-Classifiers-Based-on-Medical-Imaging](https://github.com/modabbir24/Build-COVID19-Classifiers-Based-on-Medical-Imaging)
  - Project description: We developed an advanced machine learning-based classifier that can scan chest X-rays and classify COVID19 positive cases and negative cases.

- [JunyuanLin/Capstone_Project_COVID-19_CNN_classification_based_on_Lung_CT_-Scans](https://github.com/JunyuanLin/Capstone_Project_COVID-19_CNN_classification_based_on_Lung_CT_-Scans)
  - Project description: This project aims to develop a convolutional neural network (CNN) based model to identify COVID-19 positive patients based on their lung CT scans.
  - License: [GNU General Public License (GPL) v3.0](https://github.com/JunyuanLin/Capstone_Project_COVID-19_CNN_classification_based_on_Lung_CT_-Scans/blob/master/LICENSE)

- [kabaka121212/Identify-COVID-19-from-chest-X-ray-images-by-Artificial-Intelligence](https://github.com/kabaka121212/Identify-COVID-19-from-chest-X-ray-images-by-Artificial-Intelligence)
  - Project description: Seeking the best and cheapest way to diagnose individuals with COVID-19 infections or suspects.

- [BrixIA/Brixia-score-COVID-19](https://github.com/BrixIA/Brixia-score-COVID-19)
  - Project description: Info, code (BS-Net), link to data (BrixIA COVID-19 Dataset annotated with Brixia-score), and additional material related to the BrixIA COVID-19 Project
    - Related research paper: [End-to-end learning for semiquantitative rating of COVID-19 severity on Chest X-rays](https://arxiv.org/abs/2006.04603) (Alberto Signoroni, Mattia Savardi, Sergio Benini, Nicola Adami, Riccardo Leonardi, Paolo Gibellini, Filippo Vaccher, Marco Ravanelli, Andrea Borghesi, Roberto Maroldi, Davide Farina (University of Brescia))
    - Related dataset: [BrixIA: COVID19 severity score assessment project and database](https://brixia.github.io/)

- [COVID-19-AI-Research-Project/AI-Classification](https://github.com/COVID-19-AI-Research-Project/AI-Classification)
  - Project description: Open source Artificial Intelligence for COVID-19 detection/early detection. Includes Convolutional Neural Networks (CNN) & Generative Adversarial Networks (GAN)
  - Project Facebook page: [Peter Moss Covid-19 AI Research Project](https://www.facebook.com/covid19airesearch)

- [vijay-ss/Covid19-Xray-Detection](https://github.com/vijay-ss/Covid19-Xray-Detection)
  - Project description: Covid-19 X-Ray Detection model using deep learning
  - Online implementation: https://covid19xraydetection.herokuapp.com/

- [LaurentVeyssier/Detecting-COVID-19-with-chest-X-Ray-using-PyTorch](https://github.com/LaurentVeyssier/Detecting-COVID-19-with-chest-X-Ray-using-PyTorch)
  - Project description: Deep learning model to classify XRay Scan images between 3 classes and detect COVID19

- [sumitrj/FSARL](https://github.com/sumitrj/FSARL)
  - Project description: Detection of COVID-19 from chest Xray images using Active Learning

- [sumitrj/ChestXray-TransferLearning](https://github.com/sumitrj/ChestXray-TransferLearning)
  - Project description: Plug & Play easy to use code for multi-channel transfer learning; applied for detection of COVID-19 in CXR images

- [gauravsinha7/Covid-Xray-Analysis-Tool](https://github.com/gauravsinha7/Covid-Xray-Analysis-Tool)
  - Project description: This repository contains Python code to process chest congestion XRay analysis for COVID-19.

- [ranabanik/COVID_Xray_classification](https://github.com/ranabanik/COVID_Xray_classification)
  - Project description: 2D Xray lung image, that has 3 different imbalanced classes: Covid, Pneumonia and Normal.

- [bloodbee/xray_covid_classifier](https://github.com/bloodbee/xray_covid_classifier)
  - Project description: ML Classifier to detect xray with covid disease

- [utkucolak/covid-detection-from-xray](https://github.com/utkucolak/covid-detection-from-xray)
  - Project description: Deep Learning model which tries to determine if covid-19 positive or negative according to its lung x-ray images
  - License: [Apache License 2.0](https://github.com/utkucolak/covid-detection-from-xray/blob/main/LICENSE)

- [Priyanshu-C/COVID-DETECTION-USING-XRAY](https://github.com/Priyanshu-C/COVID-DETECTION-USING-XRAY)
  - Project description: This app has mutiple models you can predict the xray upon, we trained a model on over 9 CNN models with different layers setup along with pretrained model like VGG16 and VGG19.

- [gokulnair2001/COVID-TODAY](https://github.com/gokulnair2001/COVID-TODAY)
  - Project description: COVID TODAY is an COVID determination app which uses users chest Xray to diagnose it. The app also provides live updates of worldwide cases.
  - License: [MIT License](https://github.com/gokulnair2001/COVID-TODAY/blob/main/LICENSE)

- [faniadev/covid19-xray-image-classification-using-CNN](https://github.com/faniadev/covid19-xray-image-classification-using-CNN)
  - Project description: Classifying COVID-19 X-Ray Images using Convolutional Neural Network (Tensorflow and Keras)
  - Explanation of the program and code: https://youtu.be/z0ihtCALmz4

- [abshubh/Covid-19-Detection-using-Chest-Xray-Images-through-Deep-Learning](https://github.com/abshubh/Covid-19-Detection-using-Chest-Xray-Images-through-Deep-Learning)
  - Project description: The Project was based on CNN model, which through deep learning can help detect COVID using Chest Radiography images.

- [vishal-s-v/Covid-Xray-analyser](https://github.com/vishal-s-v/Covid-Xray-analyser)
  - Project description: An application to predict COVID cases using X-ray images. Implemented using Keras and Flask API

- [Siddhant-K-code/COVID-19-RAPID-TESTER](https://github.com/Siddhant-K-code/COVID-19-RAPID-TESTER)
  - Project description: ML Model will detect the patient’s condition whether patient is positive or negative, than the person is prompted to consult a doctor.

- [cameronnunez/Diagnosing-COVID-and-pneumonia-from-chest-Xrays-Multiclass-Classification](https://github.com/cameronnunez/Diagnosing-COVID-and-pneumonia-from-chest-Xrays-Multiclass-Classification)
  - Project description: This project attempts to diagnose patients with COVID-19, viral pneumonia, and bacterial pneumonia from images of chest X-rays. The goal is to develop a multiclass classifier that achieves good weighted categorization accuracy on a set of unseen examples.

- [abhisheksakibanda/covid-19-detection-with-xrays-using-pytorch](https://github.com/abhisheksakibanda/covid-19-detection-with-xrays-using-pytorch)
  - Project description: A Resnet-18 model trained to detect COVID-19 with Chest X-Ray's

- [ShoumikMajumdar/Covid-Detection-From-Chest-XRays](https://github.com/ShoumikMajumdar/Covid-Detection-From-Chest-XRays)
  - Project description: COVID19 detection from chest X Ray Scans 

- [vk1996/COVID19_DETECTION_FROM_XRAY](https://github.com/vk1996/COVID19_DETECTION_FROM_XRAY)
  - Project description: This is a research project to analyse feasibility of putting efficient deep learning models in production for detecting COVID19 samples from xray as fast and precise as possible.

- [git-aditya-star/Covid-19-detection-using-lung-xrays](https://github.com/git-aditya-star/Covid-19-detection-using-lung-xrays)
  - Project description: COVID 19 detection using lungs x ray images

- [iamrachit/Covid-19-Detection-](https://github.com/iamrachit/Covid-19-Detection-)
  - Project description: In this Repository, Viewers will be able to see the python code for covid 19 detection using chest xray images

- [aakashratha1006/Chest-Xray-COVID-Detection-CNN](https://github.com/aakashratha1006/Chest-Xray-COVID-Detection-CNN)
  - Project description: Detecting COVID-19 in chest xray images(Input) with keras, tensorflow and deep learning - used Convolutional Neural Networks (CNN).

- [sb-robo/Covid-Detection-with-XRay-images](https://github.com/sb-robo/Covid-Detection-with-XRay-images)
  - Project description: Covid Detection with XRay images among Normal & Covid Cases

- [ulises-jeremias/aa2020unlp-covid-19-xray](https://github.com/ulises-jeremias/aa2020unlp-covid-19-xray)
  - Project description: Kaggle Challenge to Classify Covid Pneumonia Xray

- [kevbow/CNN-using-covid-pneumonia-xray-dataset](https://github.com/kevbow/CNN-using-covid-pneumonia-xray-dataset)
  - Project description: In this project, I used Convolutional Neural Network (CNN) to classify x-ray images of patients with pneumonia and healthy patients. 

- [lopezbec/COVID19_XRay_Tutorial](https://github.com/lopezbec/COVID19_XRay_Tutorial)
  - Project description: Training NN to predict COVID-19 from X-Ray images

- [vam-sin/deep-covid](https://github.com/vam-sin/deep-covid)
  - Project description: A deep learning tool for the detection of COVID-19 by analyzing Chest-Xrays

- [samuel-frankliln/covid-xray](https://github.com/samuel-frankliln/covid-xray)
  - Project description: A cnn model to detect covid 19 (educational purpose only)

- [doaa-altarawy/covid_xrays](https://github.com/doaa-altarawy/covid_xrays)
  - Project description: This project is a Web tool for the screening of COVID-19 from chest x-ray images.
  - License: [MIT License](https://github.com/doaa-altarawy/covid_xrays/blob/master/LICENSE)

- [dhruv-varshney/Covid-19-Detection-Using-Xrays](https://github.com/dhruv-varshney/Covid-19-Detection-Using-Xrays)
  - Project description: Covid19 Detection Using Xrays (Jupyter notebook)

- [SateehTeppala/DetectingCOVID19](https://github.com/SateehTeppala/DetectingCOVID19)
  - Project description: Detecting covid 19 using Radiography chest XRays

- [paul-data-science/Deep-Learning-Covid-CXR](https://github.com/paul-data-science/Deep-Learning-Covid-CXR)
  - Project description: NJIT CS677 Deep Learning Project Explainability of Covid Chest Xrays
  - License: [MIT License](https://github.com/paul-data-science/Deep-Learning-Covid-CXR/blob/main/LICENSE)

- [subhasishgosw5/Covid19-Chest-XRay-CNN](https://github.com/subhasishgosw5/Covid19-Chest-XRay-CNN)
  - Project description: A web app to determine Covid19 from Lung X-Ray Images
  - License: [GNU General Public License (GPL) v3.0](https://github.com/subhasishgosw5/Covid19-Chest-XRay-CNN/blob/master/LICENSE)

- [its-ash/covid-19-x-ray](https://github.com/its-ash/covid-19-x-ray)
  - Project description: Covid Detection Using Xray

- [Dasharath4812/Covid-19-detection-using-chest-X-rays](https://github.com/Dasharath4812/Covid-19-detection-using-chest-X-rays)
  - Project description: Classifying Xrays into covid 19 positive and normal Xrays

- [muhammedtalo/COVID-19](https://github.com/muhammedtalo/COVID-19)
  - Project description: Automated Detection of COVID-19 Cases Using Deep Neural Networks with X-Ray Images
    - Related research paper: [Automated Detection of COVID-19 Cases Using Deep Neural Networks with X-ray Images](https://www.researchgate.net/publication/340935440_Automated_Detection_of_COVID-19_Cases_Using_Deep_Neural_Networks_with_X-ray_Images) (Tulin Ozturk, Muhammed Talo, Eylul Azra Yildirim, Ulas Baran Baloglu, Ozal Yildirim, U. Rajendra Acharya)

- [btrixtran/XrayChestCOVID19](https://github.com/btrixtran/XrayChestCOVID19)
  - Project description: Based on DarkCovidNet classifier which implemented for the you only look once (YOLO) real time object detection system.

- [armiro/COVID-CXNet](https://github.com/armiro/COVID-CXNet)
  - Project description: Diagnosing COVID-19 in Frontal Chest X-ray Images using Deep Learning.
    - Related research paper: [COVID-CXNet: Detecting COVID-19 in Frontal Chest X-ray Images using Deep Learning](https://arxiv.org/abs/2006.13807) (Arman Haghanifar, Mahdiyar Molahasani Majdabadi, Younhee Choi, S. Deivalakshmi, Seokbum Ko)
  - License: [MIT License](https://github.com/armiro/COVID-CXNet/blob/master/LICENSE)

- [khaqanashraf/covid19-chest-xray](https://github.com/khaqanashraf/covid19-chest-xray)
  - Project description: This repository contains code and report for the COVID19 Chest XRay image classifications by using Deep Learning techniques.

- [isabelhssilva/CovidDeepLearningXRay](https://github.com/isabelhssilva/CovidDeepLearningXRay)
  - Project description: A study on Deep Learning techniques applied to the diagnosis of Covid-19, through x-ray images.

- [shamiktiwari/Covid19_Xray](https://github.com/shamiktiwari/Covid19_Xray)
  - Project description: Covid 19 detection using X-Ray Images

- [amrithc/covid19_xray_detection](https://github.com/amrithc/covid19_xray_detection)
  - Project description: COVID19 detection using Chest X-Ray Detection

- [whosethere/dnn_xray_covid19](https://github.com/whosethere/dnn_xray_covid19)
  - Project description: Presentation of the solution to the problem in the automatic diagnosis of lung changes (covid-19 / pneumonia) on X-rays. Work done during the hackathon "Hack The Crisis".
    - Video presentation of the project: https://www.youtube.com/watch?v=I_stdPAymWU

- [shubhamkrjain/covid19-xray](https://github.com/shubhamkrjain/covid19-xray)
  - Project description: covid-19 x-ray detection using RESNET18

- [umangdubey/Detecting_Covid19_from_Xray_image](https://github.com/umangdubey/Detecting_Covid19_from_Xray_image)
  - Project description: Detection of Covid19 from Chest Xray images

- [bakharia/Covid19XRay](https://github.com/bakharia/Covid19XRay)
  - Project description: Image classification of Chest X Rays in one of three classes: Normal, Viral Pneumonia, COVID-19.

- [ajinabraham123/COVID19-using-Xray-](https://github.com/ajinabraham123/COVID19-using-Xray-)
  - Project description: COVID-19 X-ray classification using Keras (Sequential and U-Net)

- [IrisW9527/CovidXRayDeepLearning](https://github.com/IrisW9527/CovidXRayDeepLearning)
  - Project description: Measuring Convolutional Neural Network model accuracies with the latest COVID-19 XRay image data.

- [shubhamt10/Covid19xray_detection](https://github.com/shubhamt10/Covid19xray_detection)
  - Project description: Covid 19 detector from patient's chest xray. Built using Keras in Tensorflow.

- [Sujeeth-Shetty/covid19-XRay-pytorch](https://github.com/Sujeeth-Shetty/covid19-XRay-pytorch)
  - Project description: Detecting COVID-19 with Chest X-Ray using PyTorch

- [makhloufi200/covid19-xray-detect](https://github.com/makhloufi200/covid19-xray-detect)
  - Project description: This repository aims to show a simple example of a machine learning project using Keras. The program determines whether or not the image is infected with coronavirus.

- [spapazov/covid19-xray-diagnoser](https://github.com/spapazov/covid19-xray-diagnoser)
  - Project description: This project leverages computer vision and convolutional neural networks to diagnose COVID-19 patients based on X-Ray scans of their lungs, with the hope of providing alternative ways to achieve rapid widespread testing.

- [hcho22/COVID19_Xray_fastai](https://github.com/hcho22/COVID19_Xray_fastai)
  - Project description: Detecting COVID19 by analyzing X rays using CNN

- [zeeshanahmad10809/covid19-xray-pytorch](https://github.com/zeeshanahmad10809/covid19-xray-pytorch)
  - Project description: Covid19 x-ray classification using mnasnet and alexnet in pytorch.

- [hcho22/COVID19_Xray_Keras](https://github.com/hcho22/COVID19_Xray_Keras)
  - Project description: Detecting COVID19 by analyzing X rays using CNN

- [k4rth33k/covid-xray](https://github.com/k4rth33k/covid-xray)
  - Project description: An auto-retraining powered REST API with automated training pipeline for classifying XRay images as 'COVID' or 'NOT'. Built during quarantine!

- [swagatamr/covid19-chest-xray-classification-using-transfer-learning](https://github.com/swagatamr/covid19-chest-xray-classification-using-transfer-learning)
  - Project description: XRAY classification using different transfer learning model

- [Aman9026/Predict-COVID-19](https://github.com/Aman9026/Predict-COVID-19)
  - Project description: Predicts COVID-19 from PA view of X-ray when submitted on the website
  - License: [MIT License](https://github.com/Aman9026/Predict-COVID-19/blob/master/LICENSE)

- [Mjrovai/covid19Xray](https://github.com/Mjrovai/covid19Xray)
  - Project description: Covid-19 vs Pneumo Xray Detection using TensorFlow

- [mdalmas/covid19_xray_detection](https://github.com/mdalmas/covid19_xray_detection)
  - Project description: Detecting Covid 19 in a person using PA Chest X-ray images, Using Deep-learning & Tensorflow

- [haruiz/COVID19-Xray](https://github.com/haruiz/COVID19-Xray)
  - Project description: Pytorch COVID-19 Detector using X-Ray images

- [dogydev/COVID-Efficientnet-Pytorch](https://github.com/dogydev/COVID-Efficientnet-Pytorch)
  - Project description: COVID-19 detection though Xray data and the Efficientnet AutoML architecture

- [Parag0506/ChecXray](https://github.com/Parag0506/ChecXray)
  - Project description: A flutter app to detect Covid-19 disease using chest radiographs.
  - License: [GNU Affero General Public License (AGPL) v3.0](https://github.com/Parag0506/ChecXray/blob/master/LICENSE)

- [Tessium/covid19-xray-detection](https://github.com/Tessium/covid19-xray-detection)
  - Project description: Project was designed for anomaly detection from X-Ray of lungs, in order to detect case of COVID-19 or another disease.

- [devindatt/covid19-chest_xrays-analysis](https://github.com/devindatt/covid19-chest_xrays-analysis)
  - Project description: Trained a deep learning model using Keras and TensorFlow to predict COVID-19 in chest X-rays of patients.

- [hakantekgul/COVID-19_Classification](https://github.com/hakantekgul/COVID-19_Classification)
  - Project description: A classifier to detect whether a patient has COVID 19 virus or not from chest X-Ray images.

- [prady123/COVID19](https://github.com/prady123/COVID19)
  - Project description: COVID19 Classification using Xray images

- [Thejesh-M/covid19-detection-using-chest-Xray](https://github.com/Thejesh-M/covid19-detection-using-chest-Xray)
  - Project description: Using Convolutional Neural Network, I have implemented a classifier which detects whether the person is Normal or infected by other diseases(especially COVID19).

- [naveeen684/Covid19--Xray-Interpretable-Machine-Learning-](https://github.com/naveeen684/Covid19--Xray-Interpretable-Machine-Learning-)
  - Project description: Covid19 Prediction using X-RAY with LIME Explanation
  - License: [MIT License](https://github.com/naveeen684/Covid19--Xray-Interpretable-Machine-Learning-/blob/master/LICENSE)

- [Aadityapritam/Covid_19_Xray_image_pediction](https://github.com/Aadityapritam/Covid_19_Xray_image_pediction)
  - Project description: Identification of corona using X-ray image (CNN)

- [du00d/XRAY-IMAGING](https://github.com/du00d/XRAY-IMAGING)
  - Project description: Analyzing X-Ray Imaging of potential covid-patient using deep learning

- [Auggen21/Covid-Detection-using-Lung-Xray-Images](https://github.com/Auggen21/Covid-Detection-using-Lung-Xray-Images)
  - Project description: Covid19, Pnemonia, or Normal Detection using Lung Xray Images

- [inonwir/Machine-Learning-Image-Classification-Covid19-ChestXray](https://github.com/inonwir/Machine-Learning-Image-Classification-Covid19-ChestXray)
  - Project description: Machine Learning - Image classification by using dataset of chest x-ray images.

- [tsikup/covid19-xray-cnn](https://github.com/tsikup/covid19-xray-cnn)
  - Project description: A CNN model for predicting COVID19 from X-ray chest images.

- [probayes/Covid19Xray](https://github.com/probayes/Covid19Xray)
  - Project description: Classification of chest X-ray images of patients (COVID-19, pneumonia, healthy). Illustration of the interest and limits of methods which can be used to explain the decisions of CNN models in a medical context (explainability methods).
  - License: [MIT License](https://github.com/probayes/Covid19Xray/blob/master/LICENSE)

- [Dibyanshu-gtm/COVID19Xray](https://github.com/Dibyanshu-gtm/COVID19Xray)
  - Project description: Comparing X ray Images of COVID-19 Patients with Normal X rays

- [R-Mishra/covid19-xray](https://github.com/R-Mishra/covid19-xray)
  - Project description: A classification model that predicts whether an x-ray image was taken from a subject who had Coved-19, viral pneumonia or neither.

- [hollowcodes/covid19-chest-xray-classification](https://github.com/hollowcodes/covid19-chest-xray-classification)
  - Project description: An attempt to classify x-ray images of healthy patients, patients with Covid-19 (/SARS-CoV-2) and patients with other pneumonia types.

- [mdelrosa/sta208-covid19-xray](https://github.com/mdelrosa/sta208-covid19-xray)
  - Project description: Chest X-Ray classification of COVID-19 patients

- [maansisrivastava/Covid19-Chest-Xray-Model](https://github.com/maansisrivastava/Covid19-Chest-Xray-Model)
  - Project description: Model to predict Covid-19 positive or negative from Chest X-ray

- [Ashlo/Covid19-xray-Flask-app](https://github.com/Ashlo/Covid19-xray-Flask-app)
  - Project description: Covid 19 X ray Prediction

- [hakimnasaoui/COVID19-Detection-From-Chest-Xray-Images](https://github.com/hakimnasaoui/COVID19-Detection-From-Chest-Xray-Images)
  - Project description: Detecting COVID-19 in X-ray images with Convolutional Neural Networks

- [arnav8/COVID19-Detection-using-keras](https://github.com/arnav8/COVID19-Detection-using-keras)
  - Project description: Model classifies a patient's Chest XRay image as Coronavirus positive or negative

- [soumakpoddar/Covid-Detection-Using-Chest-Xray-Scans](https://github.com/soumakpoddar/Covid-Detection-Using-Chest-Xray-Scans)
  - Project description: CNN based model in Keras, and RNN model

- [prabhat-123/Detecting-Covid-19-In-X-Ray-Image](https://github.com/prabhat-123/Detecting-Covid-19-In-X-Ray-Image)
  - Project description: This repository contains a Flask app that is capable of detecting Covid19 cases from xray images.
  - License: [MIT License](https://github.com/prabhat-123/Detecting-Covid-19-In-X-Ray-Image/blob/master/LICENSE)

- [sohamsshah/Coronavirus-Prediction-from-XRay-Images](https://github.com/sohamsshah/Coronavirus-Prediction-from-XRay-Images)
  - Project description: COVID-19 prediction by using Deep Convolutional Neural Networks trained on Xray Dataset. This model is based on VGG16 Transfer Learning Technique.

- [aviralchharia/COVID-19](https://github.com/aviralchharia/COVID-19)
  - Project description: Research Project for Detection of COVID-19 from X-Ray using Deep Learning methods. Implemented convolutional neural network for classification of X-Ray Images into COVID & non-COVID cases.
  - License: [MIT License](https://github.com/aviralchharia/COVID-19/blob/master/LICENSE)

- [dishachauhan2699/Covid_19_Xray](https://github.com/dishachauhan2699/Covid_19_Xray)
  - Project description: Covid-19 xray classification

- [EXJUSTICE/COVID19_Detection_Transfer_Learning_VGG16](https://github.com/EXJUSTICE/COVID19_Detection_Transfer_Learning_VGG16)
  - Project description: Detecting COVID-19 induced Pneumonia from Chest X-rays with Transfer Learning
    - Related blog post: [Detecting COVID-19 induced Pneumonia from Chest X-rays with Transfer Learning: An implementation in Tensorflow and Keras.](https://towardsdatascience.com/detecting-covid-19-induced-pneumonia-from-chest-x-rays-with-transfer-learning-an-implementation-311484e6afc1)

- [sam-98/CovidDetectionXRay](https://github.com/sam-98/CovidDetectionXRay)
  - Project description: A program to check the covid-19 infection in humans based on chest the x-ray using keras. 

- [pimonteiro/Covid-19-Detector](https://github.com/pimonteiro/Covid-19-Detector)
  - Project description: Covid-19 XRay Detector

- [renjmindy/COVID19-XRayPneumoniaClassifier](https://github.com/renjmindy/COVID19-XRayPneumoniaClassifier)
  - Project description: CNN - object detection, classification & various model tuning for prediction optimization
    - Related blog post: [Diagnosis of COVID-19 alike Viral Pneumonia](https://renjmindy.github.io/diagnosis_of_covid-19_alike_viral_pneumonia)

- [Ashish9914/Covid-detection-using-ChestXrays](https://github.com/Ashish9914/Covid-detection-using-ChestXrays)
  - Project description: Detecting covid using chest Xray

- [SirMalamute/CovidXrays](https://github.com/SirMalamute/CovidXrays)
  - Project description: A ML Model that can detect if a person has covid based on their xrays.

- [IrisW9527/CovidXRayDeepLearning](https://github.com/IrisW9527/CovidXRayDeepLearning)
  - Project description: Measuring Convolutional Neural Network model accuracies with the latest COVID-19 XRay image data. There are only about two hundred image data which can cause the model overfitting. This is only a demo for feeding image data and tuning hyperparameters in the CNN model.

- [shubhamt10/Covid19xray_detection](https://github.com/shubhamt10/Covid19xray_detection)
  - Project description: Covid 19 detector from patient's chest xray. Built using Keras in Tensorflow.

- [mcagriaksoy/COVID-19_Detector_X-RAY](https://github.com/mcagriaksoy/COVID-19_Detector_X-RAY)
  - Project description: CNN Based COVID-19 Detector via Chest X-Ray images

- [hakantekgul/COVID-19_Classification](https://github.com/hakantekgul/COVID-19_Classification)
  - Project description: A classifier to detect whether a patient has COVID 19 virus or not from chest X-Ray images.

- [wahabrind/Covid-19_X-ray_classification](https://github.com/wahabrind/Covid-19_X-ray_classification)
  - Project description: This repository contains data set for covid 19 and code for training covid 19 xray classification plus prediction

- [luchonaveiro/covid-19-xray](https://github.com/luchonaveiro/covid-19-xray)
  - Project description: Detecting COVID-19 on X-ray images using Tensorflow 2
    - Related blog post: [Detecting COVID-19 with X-ray images and TensorFlow](https://medium.com/analytics-vidhya/detecting-covid-19-with-x-ray-images-and-tensorflow-fd3d2302bb6)

- [Shintaki/Covid19-Xray-Classification](https://github.com/Shintaki/Covid19-Xray-Classification)
  - Project description: Classifying Covid19 positive cases from Xray images 

- [dsapandora/basic_covid_19_xray](https://github.com/dsapandora/basic_covid_19_xray)
  - Project description: Training our COVID-19 detector with Keras and TensorFlow
  - License: [MIT License](https://github.com/dsapandora/basic_covid_19_xray/blob/master/LICENSE)

- [hongsea/viewmyai-covid-xray](https://github.com/hongsea/viewmyai-covid-xray)
  - Project description: We use the x ray image to predict the covid 19
  - License: [MIT License](https://github.com/hongsea/viewmyai-covid-xray/blob/master/LICENSE)

- [Impactech/covid_xray_prediction](https://github.com/Impactech/covid_xray_prediction)
  - Project description: Using deep learning to predict COVID-19 from chest X-ray images

- [nansravn/xray-covid19](https://github.com/nansravn/xray-covid19)
  - Project description: In this tutorial, instead of using TensorFlow and Keras, Azure Custom Vision is being used as the engine for training the Image Classification model.
  - License: [MIT License](https://github.com/nansravn/xray-covid19/blob/master/LICENSE)

- [phaneendrakumarcv/covid19detectorcnn](https://github.com/phaneendrakumarcv/covid19detectorcnn)
  - Project description: A simple CNN Model that can be used to predict if a person is infected with COVID19 using Xrays of Chest

- [famunir/covid19-detection-using-Xray-images](https://github.com/famunir/covid19-detection-using-Xray-images)
  - Project description: Detection of covid19 individuals using X-ray images

- [AshuMaths1729/COVID-19_XRay_Classifier](https://github.com/AshuMaths1729/COVID-19_XRay_Classifier)
  - Project description: The project aims to predict if a person is Normal or having normal Pneumonia or is suffering from COVID-19 Pneumonia.

- [AlexoDz/ai_covid_app](https://github.com/AlexoDz/ai_covid_app)
  - Project description: Flutter AI COVID-19 Detection using Chest-xray images application

- [fral8/InceptionV3COVID19](https://github.com/fral8/InceptionV3COVID19)
  - Project description: This research repository aims to share a methodology on the use of deep learning for XRAY lungs images in order to predict COVID, Pneumonia or Normal classes
  - License: [Apache License 2.0](https://github.com/fral8/InceptionV3COVID19/blob/master/LICENSE)

- [soumitra9/Covid-19-Detection-from-chest-Xrays](https://github.com/soumitra9/Covid-19-Detection-from-chest-Xrays)
  - Project description: VGG16 model for detecting Covid-19
  
- [yeohyuyong/covid-19-xray-classifier](https://github.com/yeohyuyong/covid-19-xray-classifier)
  - Project description: Covid-19 X-Ray Classifier

- [rro2q2/COVID-19_XRay_CNN](https://github.com/rro2q2/COVID-19_XRay_CNN)
  - Project description: Using a CNN to detect COVID-19 in X-Ray Scans
  
- [turahul/ChestXrayCovidDetector](https://github.com/turahul/ChestXrayCovidDetector)
  - Project description: COVID-19-CHEST-X-RAY-DETECTOR
  - License: [GNU General Public License (GPL) v3.0](https://github.com/turahul/ChestXrayCovidDetector/blob/master/LICENSE)

- [Pranav63/COVID19_Detection](https://github.com/Pranav63/COVID19_Detection)
  - Project description: This is an *educational* repo, just to try hands on, how to detect a covid +ve case from a Xray image. Implemented using TF, using VGG16 as the model for transfer learning.

- [saktheeswaranswan/covid-19-pneumonia-chest--xray-image-classification](https://github.com/saktheeswaranswan/covid-19-pneumonia-chest--xray-image-classification)
  - Project description: Pneumonia Diagnosis using XRays

- [masoudhassani/chest_xray_covid_19](https://github.com/masoudhassani/chest_xray_covid_19)
  - Project description: Training a neural network model to detect covid-19 from chest X-rays
  - License: [GNU General Public License (GPL) v3.0](https://github.com/masoudhassani/chest_xray_covid_19/blob/master/LICENSE)

- [lukerschwan/Covid-19_Lung_Image_CNN](https://github.com/lukerschwan/Covid-19_Lung_Image_CNN)
  - Project description: This project presents a high resolution neural network to predict the presence of Covid-19 in chest xrays.
  - License: [Apache License 2.0](https://github.com/lukerschwan/Covid-19_Lung_Image_CNN/blob/master/LICENSE)

- [billmono95/CNN-covid19-detection-xray](https://github.com/billmono95/CNN-covid19-detection-xray)
  - Project description: A CNN that try to predict from an Xray if you are effect by COVID or by Pneumonia

- [nigelhussain/Covid-19_Xray_Detection](https://github.com/nigelhussain/Covid-19_Xray_Detection)
  - Project description: Repository for detecting covid-19 in x-rays

- [cvillad/covid_xray_model](https://github.com/cvillad/covid_xray_model)
  - Project description: Neural network for image classification

- [shashwatwork/COVID-19-Chest-XRay-detection](https://github.com/shashwatwork/COVID-19-Chest-XRay-detection)
  - Project description: COVID-19 Chest XRAY Detection using Streamlit

- [onvungocminh/Covid-19Xray](https://github.com/onvungocminh/Covid-19Xray)
  - Project description: This project is implemented to detect covid-19 by using Xray images.

- [11fenil11/Covid19-Detection-Using-Chest-X-Ray](https://github.com/11fenil11/Covid19-Detection-Using-Chest-X-Ray)
  - Project description: Covid-19 detection in chest x-ray images using Convolution Neural Network.
  - License: [MIT License](https://github.com/11fenil11/Covid19-Detection-Using-Chest-X-Ray/blob/master/LICENSE)

- [SaikrishnaPulipati533/Covid19-XrayImages-DL-TerraAI-Project](https://github.com/SaikrishnaPulipati533/Covid19-XrayImages-DL-TerraAI-Project)
  - Project description: Identify covid19 x-ray images using keras, tensorflows

- [itsvivekghosh/Detecting-COVID-19-using-Xray](https://github.com/itsvivekghosh/Detecting-COVID-19-using-Xray)
  - Project description: Detect Disease using Deep Learning and Transfer Learning(ResNet50)
  - License: [GNU General Public License (GPL) v3.0](https://github.com/itsvivekghosh/Detecting-COVID-19-using-Xray/blob/master/LICENSE)

- [KhizarAziz/covid-detector](https://github.com/KhizarAziz/covid-detector)
  - Project description: A multi class classifier based on fastai for xray classification and covid19 detection.

- [saisrirammortha/Covid-Chest-Xray-Classification](https://github.com/saisrirammortha/Covid-Chest-Xray-Classification)
  - Project description: Classify Chest X-ray Images of patients into COVID-Positive and Negative

- [DivyamSharma04/Covid-19-ChestXray-Prediction](https://github.com/DivyamSharma04/Covid-19-ChestXray-Prediction)
  - Project description: Covid Prediction Through Chest X ray

- [shashilsravan/codeintine](https://github.com/shashilsravan/codeintine)
  - Project description: covid 19 detector with xray images

- [arpit1920/COVID-19-Xray-Detection](https://github.com/arpit1920/COVID-19-Xray-Detection)
  - Project description: Detecting COVID-19 in X-ray images with Keras, TensorFlow, and Deep Learning
  - License: [MIT License](https://github.com/arpit1920/COVID-19-Xray-Detection/blob/master/LICENSE)

- [daniel-obare/Chest-Xray-on--COVID-19-detection](https://github.com/daniel-obare/Chest-Xray-on--COVID-19-detection)
  - Project description: Chest Analysis to detect X-ray image with COVID-19

- [K-adu/covid19-detection-from-xray](https://github.com/K-adu/covid19-detection-from-xray)
  - Project description: Covid detection using neuralnet

- [daniel-obare/Chest-Xray-on--COVID-19-detection](https://github.com/daniel-obare/Chest-Xray-on--COVID-19-detection)
  - Project description: Chest Analysis to detect X-ray image with COVID-19

- [OrtizThiago/Licon-Xray](https://github.com/OrtizThiago/Licon-Xray)
  - Project description: This repository aim to manage the creation of a CNN to auxiliate covid-19 diagnosis
  - License: [MIT License](https://github.com/OrtizThiago/Licon-Xray/blob/master/LICENSE)

- [l-kwok/COVID-19-XRAY-CNN](https://github.com/l-kwok/COVID-19-XRAY-CNN)
  - Project description: Using Artificial Intelligence (Convolutional Neural Network) to detect COVID-19 in chest x-rays

- [otsapp/covid-xray-classifier](https://github.com/otsapp/covid-xray-classifier)
  - Project description: Uses pre-trained resnet-18 to classify x-ray images with developed covid-19 vs. other lung conditions

- [jeevankishorekn/Covid-19-Deep-Learning-CNN](https://github.com/jeevankishorekn/Covid-19-Deep-Learning-CNN)
  - Project description: A Deep Learning Model created using Convolution Neural Networks to detect Covid-19 by analysing a Chest XRay.

- [deepakkvresearch/covid-detection-from-xrays](https://github.com/deepakkvresearch/covid-detection-from-xrays)
  - Project description: We present CovidAID (Covid AI Detector), a PyTorch (python3) based implementation, to identify COVID-19 cases from X-Ray images. 

- [ArkaprabhaChakraborty/ChestXrayCOVID-19](https://github.com/ArkaprabhaChakraborty/ChestXrayCOVID-19)
  - Project description: Using chest xray images to identify COVID-19 patients

- [DrDavidEllison/COVID_19_Xray](https://github.com/DrDavidEllison/COVID_19_Xray)
  - Project description: Deep learning classifier to identify covid-19 cases from X-ray images

- [Arksyd96/covid19-detection-with-xray-image-keras](https://github.com/Arksyd96/covid19-detection-with-xray-image-keras)
  - Project description: COVID-19 detection neural networik using X-ray images dataset and keras
  - License: [MIT License](https://github.com/Arksyd96/covid19-detection-with-xray-image-keras/blob/master/LICENSE)

- [subhrangshu/covid-19-xray](https://github.com/subhrangshu/covid-19-xray)
  - Project description: Classification of Covid-19 vs Normal with Chest X-Ray and Transfer Learning
  - License: [MIT License](https://github.com/subhrangshu/covid-19-xray/blob/master/LICENSE)

- [AyobamiAdebesin/Identifying-COVID19-from-Xray-](https://github.com/AyobamiAdebesin/Identifying-COVID19-from-Xray-)
  - Project description: An attempt to use Keras to detect covid-19 from X-ray images

- [abr-98/Covid-19_x_ray_detection](https://github.com/abr-98/Covid-19_x_ray_detection)
  - Project description: Covid-19 Detection using CNN using Chest Xray
  - License: [MIT License](https://github.com/abr-98/Covid-19_x_ray_detection/blob/master/LICENSE)

- [chetanpopli/covid_detection_using_xray](https://github.com/chetanpopli/covid_detection_using_xray)
  - Project description: A program to predict whether a person is covid-19 positive or negative on the basis of their chest X-ray, using machine learning

---

**NOTE**

This list focuses on a specific topic: image-based diagnosis of COVID-19.

For a broader range of COVID-19 projects and other information, I recommend lists such as [Open-Source-Covid-19](http://open-source-covid-19.weileizeng.com/) and the [Coronavirus Tech Handbook](https://coronavirustechhandbook.com/).
