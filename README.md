# Machine Learning Based Pathological Assessment with Homomorphic Encryption

## Overview

This project presents a **privacy-preserving machine learning pipeline** for pathological assessment using **Support Vector Machine (SVM)** and **Fully Homomorphic Encryption (FHE)**.

The system enables secure prediction on encrypted medical data, allowing a client to obtain pathological assessment results without revealing sensitive patient information to the server.

The project integrates:

- Machine Learning based classification
- SVM models with different kernels
- Autoencoder-based feature extraction for medical images
- CKKS-based Fully Homomorphic Encryption
- Secure encrypted inference using OpenFHE


---

## Problem Statement

Medical datasets contain highly sensitive patient information. Traditional machine learning approaches require sending raw patient data to a server, creating privacy risks.

This project addresses the problem by enabling:

- Client-side data encryption
- Server-side encrypted prediction
- No exposure of original patient records
- Privacy-preserving medical assessment


---

# System Architecture
<img width="8192" height="1541" alt="ML-Encryption" src="https://github.com/user-attachments/assets/bf90b973-6474-4154-95d4-9eb696b0726b" />

## A) Tabular Dataset Pipeline (Direct Feature Training)

### Workflow

1. Tabular medical features are used to train an SVM classifier.
2. The trained SVM model is deployed on the server.
3. The client encrypts the patient record using CKKS encryption.
4. The server performs prediction directly on encrypted data.
5. The encrypted prediction result is returned to the client.

## B) Image Dataset Pipeline with Autoencoder Feature Extraction

### Workflow

1. Medical images are passed through an autoencoder.
2. The autoencoder learns a compact representation of high-dimensional images.
3. Extracted features are used to train an SVM classifier.
4. Client encrypts medical input using CKKS encryption.
5. The server performs encrypted inference.
6. The encrypted result is sent back securely.


---

# Key Features

## Privacy-Preserving Inference

- Patient records remain encrypted during prediction.
- Server never accesses raw medical information.
- Uses CKKS Fully Homomorphic Encryption scheme.


## Machine Learning Model

Implemented and evaluated:

- Support Vector Machine (SVM)
- Multiple kernel approaches
- Feature-based classification


## Autoencoder Feature Extraction

Used for medical imaging data:

- Reduces feature dimensionality
- Learns meaningful representations
- Improves encrypted computation efficiency


## Homomorphic Encryption

Technology:

- CKKS scheme
- OpenFHE library
- 128-bit security level

Allows:

- Computation directly on encrypted data
- Secure prediction without decryption


---

# Performance Results

Performance Results

1. Benchmark Dataset:
   Model Used: SVM + Homomorphic Encryption
   Achieved F1-Score: Up to 0.96


2. Medical Imaging Dataset:
   Model Used: Autoencoder + SVM + Homomorphic Encryption
   Achieved F1-Score: Up to 0.85


Encrypted inference:

- Maintained performance close to plaintext models
- Reduced prediction latency using polynomial approximation


---

# Optimization Technique

## Chebyshev Polynomial Approximation

Non-linear kernel computation is optimized using Chebyshev polynomial approximation.

Benefits:

- Enables efficient computation under FHE
- Reduces encrypted inference latency
- Improves practical deployment feasibility


---

# Technology Stack

## Programming

- C++
- Python


## Machine Learning

- Scikit-learn
- Support Vector Machine
- Autoencoders


## Privacy & Security

- OpenFHE
- CKKS Fully Homomorphic Encryption


## Concepts

- Machine Learning
- Secure Inference
- Privacy Preserving AI
- Feature Extraction
- Encrypted Computation

---

# How It Works

### Client Side

1. Patient data is collected.
2. Data is encrypted using CKKS.
3. Encrypted data is sent to server.
4. Client receives encrypted prediction.


### Server Side

1. Trained SVM model is loaded.
2. Encrypted input is processed.
3. Prediction is performed without decryption.
4. Encrypted output is returned.


# Applications

- Privacy-preserving healthcare AI
- Secure medical diagnosis
- Cloud-based pathological assessment
- Confidential patient analytics

---

