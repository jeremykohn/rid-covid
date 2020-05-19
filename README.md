# Radiological Image Data for Clinical Open-source Viral Infection Diagnosis (RID-COVID)

A list of software projects, datasets, and other publicly available resources for diagnosing COVID-19 based on clinical images such as X-rays and CT scans.

For now, the goal is to help people find what already exists. Eventually this effort may develop into a project to produce new diagnostic tools suitable for clinical use.


## Software Projects and Tutorials

- [JordanMicahBennett/SMART-CT-SCAN_BASED-COVID19_VIRUS_DETECTOR](https://github.com/JordanMicahBennett/SMART-CT-SCAN_BASED-COVID19_VIRUS_DETECTOR)
  - Description: An image based Xray attempt at coronavirus2019 (covid19) diagnosis using a convolutional neural network design.
  - License: [MIT License](https://github.com/JordanMicahBennett/SMART-CT-SCAN_BASED-COVID19_VIRUS_DETECTOR/blob/master/LICENSE.txt)

- [lindawangg/COVID-Net](https://github.com/lindawangg/COVID-Net)
  - Description: COVID-Net Open Source Initiative
    - Related research paper: [COVID-Net: A Tailored Deep Convolutional Neural Network Design for Detection of COVID-19 Cases from Chest Radiography Images](https://arxiv.org/abs/2003.09871) (Linda Wang, Zhong Qiu Lin, and Alexander Wong; Department of Systems Design Engineering, University of Waterloo, Canada; Waterloo Artificial Intelligence Institute, Canada; DarwinAI Corp., Canada)
    - Related dataset: [agchung/Figure1-COVID-chestxray-dataset](https://github.com/agchung/Figure1-COVID-chestxray-dataset)
  - License: [GNU Affero General Public License 3.0](https://github.com/lindawangg/COVID-Net/blob/master/LICENSE.md)

- [IliasPap/COVIDNet](https://github.com/IliasPap/COVIDNet)
  - Description: PyTorch implementation of [COVID-Net](https://github.com/lindawangg/COVID-Net)
  - License: [GNU General Public License 3.0](https://github.com/IliasPap/COVIDNet/blob/master/LICENSE)

- [zeeshannisar/COVID-19](https://github.com/zeeshannisar/COVID-19)
  - Description: Detecting and Visualising the Infectious Regions of COVID-19 in X-ray Images Using Different Pretrained-Networks in Tensorflow 2.x.

- [Chester the AI Radiology Assistant](https://github.com/mlmed/dl-web-xray)
  - Description: NOT FOR MEDICAL USE. This is a prototype system for diagnosing chest x-rays using neural networks.
  - Implementations: 
    - [In-browser web app](https://mlmed.org/tools/xray/)
    - [Downloadable macOS app](https://github.com/mlmed/dl-web-xray/releases/download/2.0/Chester.app.zip)

- [sydney0zq/covid-19-detection](https://github.com/sydney0zq/covid-19-detection)
  - Description: The implementation of "Deep Learning-based Detection for COVID-19 from Chest CT using Weak Label".
    - Related research paper: [Deep Learning-based Detection for COVID-19 from Chest CT using Weak Label](https://www.medrxiv.org/content/10.1101/2020.03.12.20027185v2) (Chuangsheng Zheng, Xianbo Deng, Qiang Fu, Qiang Zhou, Jiapei Feng, Hui Ma, Wenyu Liu, Xinggang Wang;
Department of Radiology, Union Hospital, Tongji Medical College, Huazhong University of Science and Technology, Wuhan, 430022, China; Hubei Province Key Laboratory of Molecular Imaging, Wuhan, 430022, China; Artificial Intelligence Institute, School of Electronic Information and Communications, Huazhong University of Science and Technology, Wuhan, 430074, China) (Preprints: [v1](https://www.medrxiv.org/content/10.1101/2020.03.12.20027185v1.full.pdf), [v2](https://www.medrxiv.org/content/10.1101/2020.03.12.20027185v2))
  - Online implementation: http://39.100.61.27/ This website provides online testing on user-provided CT volume, and the results are the probabilities of being a normal person and being infected by COVID-19.
  - License: [Creative Commons CC-BY-NC-SA-4.0](https://github.com/sydney0zq/covid-19-detection/blob/master/CC-BY-NC-SA-4.0)
  - Related GitHub Gist: [convert_dicom_to_npy.py](https://gist.github.com/sydney0zq/4813618fd92781618e3c90809fc1db8b)

- [elcronos/COVID-19](https://github.com/elcronos/COVID-19) (Predecessor of [FightCOVID19](https://github.com/FightCOVID19))
  - Description: COVID-19 Detector from x-rays using Computer Vision and Deep Learning
  - License: COVID-19 Detector by Camilo Pestana is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License.

- [FightCOVID19](https://github.com/FightCOVID19)
  - Description: Fight COVID-19 is a non-profit, collaborative community democratising AI to assist in the detection and triage of COVID-19 cases
  - Project website (with information on upcoming Deepvision app): https://www.fightcovid19.ai/

- [PyTorchLightning/lightning-Covid19](https://github.com/PyTorchLightning/lightning-Covid19)
  - Description: A detector for covid-19 chest X-ray images using PyTorch Lightning (for educational purposes)

- [Detecting COVID-19 in X-ray images with Keras, TensorFlow, and Deep Learning](https://www.pyimagesearch.com/2020/03/16/detecting-covid-19-in-x-ray-images-with-keras-tensorflow-and-deep-learning/)
  - Description: In this tutorial, you will learn how to automatically detect COVID-19 in a hand-created X-ray image dataset using Keras, TensorFlow, and Deep Learning.
  - Availability: (Source code can be downloaded upon request via email)

- [AleGiovanardi/covidhelper](https://github.com/AleGiovanardi/covidhelper)
  - Description: Detect COVID19 infection from RX and CT scans
  

- [rekalantar/covid19_detector](https://github.com/rekalantar/covid19_detector)
  - Description: Using Deep Learning to detect Covid-19 from X-Ray/CT scans of patients

- [bkong999/COVNet](https://github.com/bkong999/COVNet)
  - Description: This is a PyTorch implementation of the paper "[Artificial Intelligence Distinguishes COVID-19 from Community Acquired Pneumonia on Chest CT](https://pubs.rsna.org/doi/10.1148/radiol.2020200905)". It supports training, validation and testing for COVNet.

- [ahmed3991/Covid-19-X-Rays-Detector](https://github.com/ahmed3991/Covid-19-X-Rays-Detector)
  - Description: Detect Covid-19 infection from X-Rays

- [chiragsamal/COVID19-Detection](https://github.com/chiragsamal/COVID19-Detection)
  - Description: CoronaVirus (COVID-19) detection using X_Ray Images

- [Synthetic.Network](https://synthetic.network/)
  - Description: NOT FOR MEDICAL USE. This is a prototype of a deep learning tool to diagnose frontal chest X-ray images and recognize bacterial pneumonia, viral pneumonia and coronavirus. 

- [aildnont/covid-cxr](https://github.com/aildnont/covid-cxr)
  - Description: Neural network model for classifying chest X-rays by presence of COVID-19 features
  - License: [MIT License](https://github.com/aildnont/covid-cxr/blob/master/LICENSE)

- [velebit-ai/COVID-Next-Pytorch](https://github.com/velebit-ai/COVID-Next-Pytorch)
  - Description: COVID-Next -> Pytorch upgrade of the COVID-Net for COVID-19 detection in X-Ray images
  - License: [MIT License](https://github.com/velebit-ai/COVID-Next-Pytorch/blob/master/LICENSE)

- [manujosephv/covid-xray-imagenet](https://github.com/manujosephv/covid-xray-imagenet)
  - Description: Imagenet Pretraining for Covid-19 Xray Identification
    - Related blog post: [Does Imagenet Pretraining work for Chest Radiography Images(COVID-19)?](https://deep-and-shallow.com/2020/04/05/does-imagenet-pretraining-work-for-chest-radiography-imagescovid-19/)
  - License: [MIT License](https://github.com/manujosephv/covid-xray-imagenet/blob/master/LICENSE)

- [sagarnildass/covid_19_xray_classification](https://github.com/sagarnildass/covid_19_xray_classification)
  - Description: In this repo, we will explore the methodology of implementing AI Based solutions for comparing COVID-19 against other types of viral illness, and will analyze the results obtained.

- [adeaeede/ai4covid](https://github.com/adeaeede/ai4covid)
  - Description: This project is a result of the [WirVsVirus Hackathon](https://wirvsvirushackathon.org/). Our objective was to assist doctors in diagnosing COVID-19 patients by training a convolutional network to discriminate between patients with and without the disease, based on an X-ray image.

- [ChirilaLaura/COVID-Z](https://github.com/ChirilaLaura/COVID-Z)
  - Description: An online AI based platform for COVID-19 testing.
  - Web page on Devpost: https://devpost.com/software/covid-z
  - License: [MIT License](https://github.com/ChirilaLaura/COVID-Z/blob/master/LICENSE)

- [ChirilaLaura/COVID-X](https://github.com/ChirilaLaura/COVID-X) ([V1](https://github.com/ChirilaLaura/COVID-X), [V2](https://github.com/ChirilaLaura/COVID-X_V2), [V3](https://github.com/ChirilaLaura/COVID-X_V3))
  - License: [MIT License](https://github.com/ChirilaLaura/COVID-X/blob/master/LICENSE)

- [hananshafi/covid19-detection](https://github.com/hananshafi/covid19-detection)
  - Description: This code is for predicting COVID-19 from chest Xrays.

- [tawsifur/COVID-19-Chest-X-ray-Detection](https://github.com/tawsifur/COVID-19-Chest-X-ray-Detection)
  - Description: A team of researchers from Qatar University, Doha, Qatar and the University of Dhaka, Bangladesh along with their collaborators from Pakistan and Malaysia in collaboration with medical doctors have managed to classify COVID-19, Viral pneumonia and Normal Chest X-ray images with an accuracy of 98.3%.
    - Related research paper: [Can AI help in screening Viral and COVID-19 pneumonia?](https://arxiv.org/abs/2003.13145) (M.E.H. Chowdhury, T. Rahman, A. Khandakar, R. Mazhar, M.A. Kadir, Z.B. Mahbub, K.R. Islam, M.S. Khan, A. Iqbal, N. Al-Emadi, M.B.I. Reaz)
    - Related dataset: [COVID-19 Radiography Database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)

- [talhaanwarch/Corona_Virus](https://github.com/talhaanwarch/Corona_Virus)
  - Description: Diagnosis of corona virus using Chest Xray, through deep learning

- [defeatcovid19/defeatcovid19-net-pytorch](https://github.com/defeatcovid19/defeatcovid19-net-pytorch)
  - Description: Pytorch solution for predictions on X-ray images of COVID-19 patients
  - License: [MIT License](https://github.com/defeatcovid19/defeatcovid19-net-pytorch/blob/master/LICENSE)

- [itratrahman/covid_19](https://github.com/itratrahman/covid_19)
  - Description: Project to detect COVID19 from X-Rays.

- [saisriteja/covid19](https://github.com/saisriteja/covid19)
  - Description: COVID image data analysis using chest X-rays and CT scans.

- [BioXAI/DeepCOVIDExplainer](https://github.com/BioXAI/DeepCOVIDExplainer)
  - Description: Explainable COVID-19 Diagnosis from Chest Radiography Images

- [appushona123/Instant-COVID19-detection](https://github.com/appushona123/Instant-COVID19-detection)
  - Description: This project is able to detect the COVID19 patient with the X-ray image of the person. 

- [talhaanwarch/Corona_Virus](https://github.com/talhaanwarch/Corona_Virus)
  - Description: Diagnosis of corona virus using Chest Xray, through deep learning

## Image Data

- [ieee8023/covid-chestxray-dataset](https://github.com/ieee8023/covid-chestxray-dataset)
  - Description: We are building a database of COVID-19 cases with chest X-ray or CT images. We are looking for COVID-19 cases as well as MERS, SARS, and ARDS.
    - Related research paper: [On the limits of cross-domain generalization in automated X-ray prediction](https://arxiv.org/abs/2002.02497) (Joseph Paul Cohen; Mohammad Hashir; Rupert Brooks; Hadrien Bertrand Mila, Université de Montréal)
  - License: (Some images in this dataset are licensed under CC BY, CC BY-NC-SA, CC BY-NC-ND, or Apache 2.0; for many other images, the license is unknown)

- [ml-workgroup/covid-19-image-repository](https://github.com/ml-workgroup/covid-19-image-repository)
  - Description: This project aims to create an anonymized data set of COVID-19 cases with a focus on radiological imaging. This includes images with extensive metadata, such as admission-, ICU-, laboratory-, and patient master-data.
  - License: CC BY 3.0

- [UCSD-AI4H/COVID-CT](https://github.com/UCSD-AI4H/COVID-CT)
  - Description: The COVID-CT-Dataset has CT images containing clinical findings of COVID-19. We are continuously adding new COVID CT images and we would like to invite the community to contribute COVID CTs as well.
    - Related research paper: [COVID-CT-Dataset: A CT Scan Dataset about COVID-19](https://arxiv.org/abs/2003.13865) (Jinyu Zhao, UC San Diego; Yichen Zhang, UC San Diego; Xuehai He, UC San Diego; Pengtao Xie, UC San Diego, Petuum Inc)

- [agchung/Figure1-COVID-chestxray-dataset](https://github.com/agchung/Figure1-COVID-chestxray-dataset)
  - Description: Figure 1 COVID-19 Chest X-ray Dataset Initiative. We are building this dataset as a part of the COVIDx dataset to enhance our models for COVID-19 detection. Please see the main [COVID-Net](https://github.com/lindawangg/COVID-Net) repo for details on data extraction and instructions for creating the full COVIDx dataset.

- [coyotespike/covid19-images](https://github.com/coyotespike/covid19-images)
  - Description: A large collection of COVID-19 radiology imaging datasets for use in machine learning.

- [CORONACASES.ORG](https://coronacases.org/)
  - Description: This community is for health professionals to share confirmed cases of coronavirus. We review posts in order to assure anonymization.

- [COVID-19 BSTI Imaging Database](https://www.bsti.org.uk/training-and-education/covid-19-bsti-imaging-database/)
  - Description: The British Society of Thoracic Imaging (BSTI), in conjunction with Cimar UK’s Imaging Cloud Technology (cimar.co.uk), have designed, built and deployed a simple, free to use, anonymised and encrypted online portal to upload and refer imaging of patients with either confirmed or suspected COVID-19. From these cases, BSTI hope to provide an imaging database of known UK patient examples for reference and teaching.

- [The Role of Chest Imaging in Patient Management during the COVID-19 Pandemic](https://www.fleischner-covid19.org/)
  - Description: Welcome to the Fleischner Society’s educational repository of radiographic and CT images of patients diagnosed with COVID-19. We provide this on-line supplement to our published statement as an aid to familiarize the medical community with the typical imaging findings of COVID-19. 

- [SIRM COVID-19 DATABASE](https://www.sirm.org/category/senza-categoria/covid-19/)
  - Description: COVID-19 radiological cases from the Italian Society of Medical Radiology (SIRM), including images from X-rays and CT scans.

- [farmy-ai/covid-fighters](https://github.com/farmy-ai/covid-fighters)
  - Description: Data collection and label tool for COVID-19 disease chest scans. 
  - Project website: [COVIDEEP](http://www.covideep.net)

- [COVID-19 Radiography Database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)
  - Description: A team of researchers from Qatar University, Doha, Qatar and the University of Dhaka, Bangladesh along with their collaborators from Pakistan and Malaysia in collaboration with medical doctors have created a database of chest X-ray images for COVID-19 positive cases along with Normal and Viral Pneumonia images. 
    - Related research paper: [Can AI help in screening Viral and COVID-19 pneumonia?](https://arxiv.org/abs/2003.13145) (M.E.H. Chowdhury, T. Rahman, A. Khandakar, R. Mazhar, M.A. Kadir, Z.B. Mahbub, K.R. Islam, M.S. Khan, A. Iqbal, N. Al-Emadi, M.B.I. Reaz)
    - Code repository: [tawsifur/COVID-19-Chest-X-ray-Detection](https://github.com/tawsifur/COVID-19-Chest-X-ray-Detection)  

- [Aman9026/COVID-19-Predictor-dataset](https://github.com/Aman9026/COVID-19-Predictor-dataset)
  - Description: Predict COVID-19 by analyzing chest X-Ray images in this dataset.

---

**NOTE**

This list focuses on a specific topic: image-based diagnosis of COVID-19.

For a broader range of COVID-19 projects and other information, I recommend lists such as [Open-Source-Covid-19](http://open-source-covid-19.weileizeng.com/) and the [Coronavirus Tech Handbook](https://coronavirustechhandbook.com/).
