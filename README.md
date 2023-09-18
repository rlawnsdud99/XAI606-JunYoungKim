### XAI606 Course Project Description

---

#### I. Project Title
**Optimizing Classification Performance in Handwriting EEG Data**

---

#### II. Project Introduction

##### Objective
The primary objective of this project is to optimize the classification accuracy of EEG signals during handwriting tasks. We aim to use advanced neural network architectures and machine learning techniques to correlate EEG signals with handwriting motions, thereby achieving higher classification performance.

##### Motivation
Understanding the neural correlates of handwriting can have various applications, including but not limited to, medical diagnosis and human-computer interaction via Brain-Computer Interface (BCI). Improving the classification accuracy can significantly enhance the effectiveness of these applications.

---

#### III. Dataset Description

The dataset comprises EEG and handwriting data from 5 participants performing handwriting tasks. The data has been approved for ethical standards and involves several features, including EEG signals and handwriting trajectories.

##### Splitting the Dataset
- **Training Dataset**: 70% of the data from each participant will be used for training. It will contain both input EEG data and the corresponding ground truth of handwriting states.
  
- **Validation Dataset**: 15% of the data from each participant will be used for model validation. This subset will also include both input and ground truth.

- **Test Dataset**: The remaining 15% will be used for model testing. This set will only contain the input EEG data without any ground truth to evaluate the model's performance.

##### Release Plan
The training and validation datasets will be publicly released for project participants to train and validate their models. The test dataset will be used to evaluate the models, and only input data will be provided.

