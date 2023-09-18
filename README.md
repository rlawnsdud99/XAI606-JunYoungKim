### XAI606 Course Project Description

---

#### I. Project Title
**Optimizing Classification Performance in Handwriting EEG Data**

---

#### II. Project Introduction

##### Objective
The primary objective is to improve classification performance of EEG signals during handwriting tasks. This will be achieved through the utilization of advanced neural network architectures and machine learning techniques.

##### Motivation
The project aims to advance the field of Brain-Computer Interface (BCI) by enhancing the classification accuracy of EEG signals associated with handwriting. The improved accuracy could facilitate various applications such as medical diagnostics and enhanced human-computer interactions.

---

#### III. Dataset Description

The dataset is based on EEG and handwriting data from a single participant (P1) who performed handwriting tasks multiple times. This is considered a dependent dataset.

##### Instrumentation
- **Handwriting Data**: Collected via a self-developed Android app on a HUAWEI MatePad Pro
  - Sampling rate: 60 Hz
  - Features: x, y coordinates, timestamp, force, state codes (pen-down, pen-move, pen-up)
- **EEG Data**: Collected via a 32-channel BrainAmp amplifier
  - Sampling rate: 1000 Hz
  - Electrode placement: According to the 10–20 international system

##### Synchronization
- Key events in the handwriting stream are used as time markers in the EEG stream for synchronization.
  - S1 to S12: Represent the first pen-down events for each letter and punctuation in the sentence 'HELLO, WORLD!'
  - S20: Marks the end of the trial

##### Experimental Setup
- **Environment**: Conducted in a sound-attenuated room
- **Tablet Placement**: Landscape orientation, 35 cm sight distance, 40-degree angle above horizontal
- **Task**: Writing the sentence 'HELLO, WORLD!' 300 times in 12 identical blocks on the tablet
- **Additional**: Breaks are allowed, and P1 has conducted an extra session for testing cross-session accuracy.

##### Splitting the Dataset
- **Training Dataset**: 70% of P1's data
- **Validation Dataset**: 15% of P1's data
- **Test Dataset**: Remaining 15% of P1's data
  - Note: Only input EEG data will be provided in the test dataset for model evaluation

##### Release Plan
The training and validation datasets will be available for participants to train and verify their models. The test dataset will be solely for model evaluation.
