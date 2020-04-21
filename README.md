# RID-COVID: Radiological Image Data for Clinical Open-source Viral Infection Diagnosis

A list of software projects, datasets, and other publicly available resources for diagnosing COVID-19 based on clinical images such as X-rays and CT scans.

For now, the goal is to help people find what already exists. Eventually this effort may develop into a project to produce new diagnostic tools suitable for clinical use.


## Software Projects and Tutorials

- [SMART-CT-SCAN_BASED-COVID19_VIRUS_DETECTOR](https://github.com/JordanMicahBennett/SMART-CT-SCAN_BASED-COVID19_VIRUS_DETECTOR)
  - Description: An image based Xray attempt at coronavirus2019 (covid19) diagnosis using a convolutional neural network design.
  - License: [MIT License](https://github.com/JordanMicahBennett/SMART-CT-SCAN_BASED-COVID19_VIRUS_DETECTOR/blob/master/LICENSE.txt)

- [COVID-Net](https://github.com/lindawangg/COVID-Net)
  - Description: COVID-Net Open Source Initiative
    - Related research paper: [COVID-Net: A Tailored Deep Convolutional Neural Network Design for Detection of COVID-19 Cases from Chest Radiography Images](https://arxiv.org/abs/2003.09871) (Linda Wang, Zhong Qiu Lin, and Alexander Wong; Department of Systems Design Engineering, University of Waterloo, Canada; Waterloo Artificial Intelligence Institute, Canada; DarwinAI Corp., Canada)
  - License: [GNU Affero General Public License 3.0](https://github.com/lindawangg/COVID-Net/blob/master/LICENSE.md)

- [IliasPap/COVIDNet](https://github.com/IliasPap/COVIDNet)
  - Description: PyTorch implementation of [COVID-Net](https://github.com/lindawangg/COVID-Net)

- [zeeshannisar/COVID-19](https://github.com/zeeshannisar/COVID-19)
  - Description: Detecting and Visualising the Infectious Regions of COVID-19 in X-ray Images Using Different Pretrained-Networks in Tensorflow 2.x.

- [Chester the AI Radiology Assistant](https://github.com/mlmed/dl-web-xray)
  - Description: NOT FOR MEDICAL USE. This is a prototype system for diagnosing chest x-rays using neural networks.
  - Implementations: 
    - [In-browser web app](https://mlmed.org/tools/xray/)
    - [Downloadable macOS app](https://github.com/mlmed/dl-web-xray/releases/download/2.0/Chester.app.zip)

- [covid-19-detection](https://github.com/sydney0zq/covid-19-detection)
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
  - Website (with information on upcoming Deepvision app): https://www.fightcovid19.ai/

- [lightning-Covid19](https://github.com/PyTorchLightning/lightning-Covid19)
  - Description: A detector for covid-19 chest X-ray images using PyTorch Lightning (for educational purposes)

- [Detecting COVID-19 in X-ray images with Keras, TensorFlow, and Deep Learning](https://www.pyimagesearch.com/2020/03/16/detecting-covid-19-in-x-ray-images-with-keras-tensorflow-and-deep-learning/)
  - Description: In this tutorial, you will learn how to automatically detect COVID-19 in a hand-created X-ray image dataset using Keras, TensorFlow, and Deep Learning.
  - Availability: (Source code can be downloaded upon request via website)

- [covidhelper](https://github.com/AleGiovanardi/covidhelper)
  - Description: Detect COVID19 infection from RX and CT scans
  

- [covid19_detector](https://github.com/rekalantar/covid19_detector)
  - Description: Using Deep Learning to detect Covid-19 from X-Ray/CT scans of patients

- [COVID-19 Detection Neural Network (COVNet)](https://github.com/bkong999/COVNet)
  - Description: This is a PyTorch implementation of the paper "[Artificial Intelligence Distinguishes COVID-19 from Community Acquired Pneumonia on Chest CT](https://pubs.rsna.org/doi/10.1148/radiol.2020200905)". It supports training, validation and testing for COVNet.

- [Covid-19-X-Rays-Detector](https://github.com/ahmed3991/Covid-19-X-Rays-Detector)
  - Description: Detect Covid-19 infection from X-Rays

- [COVID19-Detection](https://github.com/chiragsamal/COVID19-Detection)
  - Description: CoronaVirus (COVID-19) detection using X_Ray Images

- [Synthetic.Network](https://synthetic.network/)
  - Description: NOT FOR MEDICAL USE. This is a prototype of a deep learning tool to diagnose frontal chest X-ray images and recognize bacterial pneumonia, viral pneumonia and coronavirus. 



## Image Data

- [covid-chestxray-dataset](https://github.com/ieee8023/covid-chestxray-dataset)
  - Description: We are building a database of COVID-19 cases with chest X-ray or CT images. We are looking for COVID-19 cases as well as MERS, SARS, and ARDS.
    - Related research paper: [On the limits of cross-domain generalization in automated X-ray prediction](https://arxiv.org/abs/2002.02497) (Joseph Paul Cohen; Mohammad Hashir; Rupert Brooks; Hadrien Bertrand Mila, Université de Montréal)
  - License: (Some images in this dataset are licensed under CC BY, CC BY-NC-SA, CC BY-NC-ND, or Apache 2.0; for many other images, the license is unknown)

- [COVID-CT](https://github.com/UCSD-AI4H/COVID-CT)
  - Description: The COVID-CT-Dataset has CT images containing clinical findings of COVID-19. We are continuously adding new COVID CT images and we would like to invite the community to contribute COVID CTs as well.
    - Related research paper: [COVID-CT-Dataset: A CT Scan Dataset about COVID-19](https://arxiv.org/abs/2003.13865) (Jinyu Zhao, UC San Diego; Yichen Zhang, UC San Diego; Xuehai He, UC San Diego; Pengtao Xie, UC San Diego, Petuum Inc)

- [Figure1-COVID-chestxray-dataset](https://github.com/agchung/Figure1-COVID-chestxray-dataset)
  - Description: Figure 1 COVID-19 Chest X-ray Dataset Initiative. We are building this dataset as a part of the COVIDx dataset to enhance our models for COVID-19 detection. Please see the main [COVID-Net](https://github.com/lindawangg/COVID-Net) repo for details on data extraction and instructions for creating the full COVIDx dataset.

---

**NOTE**

This list focuses on a specific topic: image-based diagnosis of COVID-19.

For a broader range of COVID-19 projects and other information, I recommend lists such as [Open-Source-Covid-19](http://open-source-covid-19.weileizeng.com/) and the [Coronavirus Tech Handbook](https://coronavirustechhandbook.com/).
