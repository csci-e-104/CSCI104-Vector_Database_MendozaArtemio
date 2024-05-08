# <img style="float: left; padding-right: 10px; width: 45px" src="https://raw.githubusercontent.com/Harvard-HES-ALM/master/main/ds-masters/content//images/hes-logo.png"> Harvard Extension School </br> 

## CSCI E-104: Advanced Deep Learning  
### Spring 2024

**Instructors**: Zoran B. Djordjevic & Blagoje Z. Djordjevic<br/>

**Student**: Artemio Mendoza-García</br>
<hr style="height:2pt">

# Final Project
## Vector Databases: Unveiling the Celebrities Among Us
<hr style="height:2pt">

### Why do we want to perform facial similarities against Celebrities?

It is established that aliens walk among us, as demonstrated by the documental MIB (**Men In Black, Will Smith et all. 1997**). However, little do we know if a celebrity is disguised as a regular person, living in our same neighborhood, going to the same grocery store as we do, or taking the same course, CS104-Advanced Deep Learning, and we are unaware.

With the objective to uncover the celebrities among us we built a Vector Database with more than two hundred thousand vector embeddings, corresponding to more than ten thousand 

### But, from where do we get Celebrities images?

We start with the [Dataset CELEBA, a well-known collection of images from the University of Hong Kong](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). It contains more than 200,000 images from more than 10,000 celebrities. 

For this project, we use the ["in-the-wild" images, which are available for download as a compressed file in this Google Drive](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg&usp=sharing).  

The compressed file size exceeded the 1.3 GB mark and took about one hour to decompress. To speed up creating embeddings and inserting into Milvus, the images were stored locally.


### And, how do we extract the Vector Embeddings?

We use DeepFace, which abstract embedding extraction using different CNN SOTA models:

* FaceNet
* VGG_Face (University of Oxford)
* OpenFace
* DeepFace

For this project, we used FaceNet, which Google developed in 2015. With 140 million parameters and 22-layer depth, it achieves a prediction accuracy of 99.22% on the LFW dataset. With this model, Google introduced the triplet loss function, which works by forming triplets with one anchor, positive example, and negative example. 

The image above is a high level schema of the FaceNet CNN architecture.

![Face Neet Architecture](https://raw.githubusercontent.com/csci-e-104/csci-104-FinalProject/main/images/deep-learning-architecture.png)


### What about where to store the Vector Embeddings?
We employed Milvus, an Open-source Vector Database.

Milvus can be installed locally or in the cloud. For this project, we installed it locally as a standalone instance with GPU support using the Docker image provided in the documentation manual.

The standalone instance includes three components:

Milvus: the core functional component.
Meta Store: the metadata engine that accesses and stores metadata of Internal components, including proxies, index nodes, and more.
Object Storage: The storage engine, which is responsible for data persistence for Milvus


![Face Neet Architecture](https://raw.githubusercontent.com/csci-e-104/csci-104-FinalProject/main/images/milvus_standalone_architecture.jpg)


The image above is a high level schema of the Milvus Standalone architecture

Now that we stablished the real motivations behind this project, lets get to the documentation.

---

# Facial Recognition with Vector Databases

## 1. Overview

Facial recognition is a critical aspect of many modern applications, from security systems to social media platforms. This project explores the use of vector databases for facial recognition, using the CELEBA dataset and leveraging the power of DeepFace and Milvus. 

The goal is to demonstrate how vector databases can efficiently handle and query large-scale embeddings for facial similarity analysis

## 1.1. Technology Description

The project employs several advanced technologies to tackle the challenge of facial recognition using vector databases. Each technology plays a critical role in the system's overall functionality:

### 1.1.1. CelebA Dataset

The CelebFaces Attributes Dataset (CelebA) is a richly-annotated collection of over 200,000 celebrity images, each labeled with 40 attribute tags, covering more than 10,000 different celebrities. Each image is associated with a unique ID rather than names to maintain anonymity and privacy. The dataset is widely used in academic and research settings for experiments involving facial attribute recognition, facial detection, and various other computer vision applications. Its extensive size and high variance in facial features make it ideal for training and testing advanced facial recognition systems.

### 1.1.2. DeepFace Library

DeepFace is a deep learning facial recognition system created by Facebook that dramatically reduces the error rate in face verification processes. 

For this project, the DeepFace library is utilized to handle the complex task of facial recognition by leveraging pre-trained models, including the FaceNet Convolutional Neural Network (CNN). 

FaceNet is particularly effective because it directly learns a mapping of face photos to a compact Euclidean space. This mapping clusters similar faces together, which significantly enhances the accuracy and speed of facial recognition tasks by focusing on embeddings that are computed through the CNN, transforming facial features into vector embeddings.


The image above is a high level schema of the FaceNet CNN architecture

![Face Neet Architecture](https://raw.githubusercontent.com/csci-e-104/csci-104-FinalProject/main/images/deep-learning-architecture.png)



### 1.1.3. Milvus Database

Milvus is an open-source vector database specifically designed to handle large-scale similarity search and vector indexing. In this project, Milvus stores the embeddings generated by the DeepFace library. 

It provides highly efficient storage and retrieval capabilities for high-dimensional vector data, making it perfectly suited for applications like facial recognition where quick retrieval of similar images is crucial. 

Milvus supports multiple index types and has robust capabilities for scaling both vertically and horizontally, which aids in managing the vast amounts of data processed in this project

Below, the architecture for Milvus in Standalone mode. 

![Face Neet Architecture](https://raw.githubusercontent.com/csci-e-104/csci-104-FinalProject/main/images/milvus_standalone_architecture.jpg)

### 1.1.4. OpenCV for Webcam Integration

OpenCV (Open Source Computer Vision Library) is used in this project to capture real-time images from a webcam directly within the Jupyter Notebook environment. This feature is crucial for interactive testing and demonstration purposes. 

OpenCV allows for the seamless capture and processing of live images, which can then be fed into the DeepFace model for real-time vector embeddings extraction, and then verification against the stored embeddings in the Milvus database.

## 2. Problem Statement

It is a well-established fact that aliens walk among us, as demonstrated by the documentary 'Men In Black' *(Smith, A. J., & K, A. (1997). Men in Black [Film].  Barry Sonnenfeld, Columbia Pictures.)*. 

However, little do we know if a celebrity is disguised as a regular person, living in our neighborhood, going to the same grocery store as us, or even taking the same course. We aim to use a vector database with more than 200,000 vector embeddings of over 10,000 celebrities to shed light on this potential problem.

## 2.1. Background

In an increasingly digital age, facial recognition technology has become a pivotal tool across various industries including security, entertainment, and marketing. However, the growing volume of image data poses significant challenges in terms of storage, retrieval, and comparison of facial images efficiently and accurately.

## 2.2. Motivation

The fictional premise of the project is inspired by popular culture references to covert alien presences, humorously suggesting that some celebrities might be hiding among the general population. 

Beyond the playful narrative, the real-world implication is the exploration of advanced technological solutions for identifying and categorizing faces in large datasets. This aligns with broader applications such as identity verification, crowd monitoring, and personalized marketing strategies.


## 2.3. Objective

The main objective of this project is to leverage the capabilities of vector databases to manage and retrieve large volumes of facial embeddings effectively. The project seeks to demonstrate how Deep Learning and vector database technologies like Milvus can be used to:

- Efficiently store and retrieve large sets of image embeddings.
- Quickly find similar facial embeddings from a vast database.
- Compare and identify facial features with high accuracy.

## 2.4. Challenges

he challenges identified for this project include:

Data Volume: Managing the large size and complexity of the CELEBA dataset, which contains over 200,000 images of more than 10,000 celebrities.

Performance Optimization: Ensuring the vector database is optimized for high-speed data retrieval without compromising the accuracy of facial recognition.

System Scalability: Designing the system to be scalable so that it can handle even larger datasets or potentially expand to real-time data processing in the future.

## 2.5. Hypothesis

While a formal hypotesis testing is not whitin the scope of this project, we loosly defined a hypothesis to have a guide while conducting the experiments.

The hypothesis driving this project is that a vector database, when integrated with deep learning facial recognition technologies, can significantly improve the efficiency and accuracy of image data processing compared to traditional relational database systems. 

This would provide a scalable solution that could be applied in various real-world scenarios requiring rapid and accurate facial recognition.

## 3. Dataset

### 3.1. CELEBA Overview

[The CelebFaces Attributes Dataset (CelebA)](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) is a key element in this project, providing a robust dataset for training and testing the facial recognition system.  The dataset features over 200,000 celebrity images, each annotated with 40 attribute labels. This dataset is extensively used in machine learning research, particularly in the field of facial recognition.

### 3.2. Data Characteristics

**Size and Scope**: The dataset is notably large, comprising over 10,000 in-the-wild celebrity images. Each image is tagged with attributes and associated with a unique ID rather than a name, ensuring the anonymity of the individuals. This anonymization is crucial in academic and research settings to prevent misuse of personally identifiable information.

**Format**: Images are stored in a high-quality JPEG format, ensuring that the facial features are clear and distinguishable, which is vital for the accuracy of facial recognition algorithms.

### 3.3. Data Acquisition

The dataset is publicly available for download via a [Google Drive link](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg&usp=sharing) provided by the University of Hong Kong. The download process is notably time-consuming due to the large file size; it took more than 2 hours to download the dataset. This aspect is crucial as it highlights the challenges associated with handling large-scale data in real-world applications.

### 3.4. Data Preparation

Upon downloading, the dataset was in a compressed zip format, requiring over an hour to decompress. This decompression time underscores the dataset's vastness and the computational resources required for basic data preparation steps before any actual processing or analysis can occur. Storing this data efficiently, while maintaining quick access to it, presents a significant challenge, which is addressed by using a vector database in this project.

### 3.5. Security and Privacy Considerations

Given the sensitive nature of facial data, CelebA's approach to anonymize the data by using IDs instead of names is a critical aspect. It ensures that the project adheres to ethical guidelines regarding the use of personal data in research, particularly in a field as sensitive as facial recognition. This measure is particularly important to mitigate any risks of personal data exploitation or privacy invasion.

### 3.6. Importance in the Project

In this project, CelebA serves as a foundational element, providing a robust dataset necessary for creating and testing the facial embeding Database. The vast size of the dataset enables the project to demonstrate the effectiveness of vector databases in managing large-scale data efficiently and effectively, thereby proving the hypothesis that vector databases can significantly enhance facial recognition tasks in terms of speed and accuracy.

## 4. Installation and Configuration

Note: the following sections are a minimal guide on the setup and installation of each components aimed to guide on the process I followed to implement this project, and not an extensive implementation manual on Milvus DB. For that purpose, the link to the docs are provided.

### 4.1. System Setup

The project was developed and tested on a workstation running Windows 11 with Windows Subsystem for Linux (WSL2) installed. 
The workstation sports a dual RTX 4090 GPU units, so, CUDA was installed in both, Win and WSL2 OS. 

This setup allowed for a division between the development environment and the database hosting:

**Frontend**: The Jupyter Notebook, which contains the frontend code including data processing and user interface for image capture via webcam, runs on the Windows 11 system. This approach enhances the accessibility and interactivity of the project by allowing real-time data capture and processing.

**Backend**: The backend, consisting of the Milvus vector database and Docker, operates within the WSL2 environment. This separation ensures that the database operations are handled in a more controlled and Unix-compatible environment, which is optimal for Docker and Milvus operations.

### 4.2. Conda Environment

To manage dependencies effectively and address issues related to library compatibility:

A Conda environment was created using a YAML file. This environment encapsulates all the necessary Python libraries and frameworks required for the project, ensuring that there are no conflicts between dependencies and that the project remains portable and reproducible on any compatible system.

To create the environemnt (in Win and WSL2):

1. Clone the [GitHub repository for this project](https://github.com/csci-e-104/csci-104-FinalProject):

    Using SSH

    ```bash
    git clone git@github.com:csci-e-104/csci-104-FinalProject.git
    ```

    Using HTTP

    ```bash
    git clone https://github.com/csci-e-104/csci-104-FinalProject.git
    ```

2. Create Conda (or Mamba) virtual environment

    ```bash
    conda env create -f vdb.yml 
    ```

### 4.3. Milvus Installation

Milvus is installed using Docker within WSL2, enhancing the deployment and scalability of the database.

Milvus was installed as a standalone system using Docker, which simplifies the deployment and scalability of applications. The detailed steps taken to install and manage Milvus are as follows:

**Prerequisites**
Docker Installation: Prior to installing Milvus, Docker was installed on WSL2, which is a prerequisite for running Milvus in a containerized environment.

Installation Steps

__Download Installation Script:__
```bash
# Copy code
wget https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh
```
Start Milvus:
```bash
# Copy code
bash standalone_embed.sh start
```
This script initiates the Milvus standalone version, setting up the necessary components including the storage and metadata engines.

__Connect to Milvus:__

After starting Milvus, connection setups were implemented as per the instructions in the Milvus documentation's Quickstart guide. 

This involved configuring the system to connect the Jupyter notebook interface with the Milvus backend, enabling seamless data flow between the image processing code and the database.


**Stop Milvus**:

To stop the Milvus service, the following command was used:
```bash
# Copy code
bash standalone_embed.sh stop
```
This command ensures that the database can be safely shut down without data loss.

**Delete Data**:

If necessary, to delete all data stored in Milvus after stopping the service:
```bash
# Copy code
bash standalone_embed.sh delete
```

This step is crucial for cleaning up the environment or preparing the system for a fresh start without previous data interference.


### 4.4. Webcam Integration for Testing

Interactive Testing: The project's Jupyter Notebook includes Python code that allows capturing images directly from the webcam. This feature facilitates easy and fast testing of the facial recognition system, allowing real-time interaction and demonstration of the project's capabilities.

## 5. Experiments

The experiments were designed to validate the effectiveness of the vector database, Milvus, in handling and retrieving facial embeddings derived from the CELEBA dataset. These experiments utilized DeepFace to extract facial features and embed them into the Milvus database. 

The Jupyter Notebook [```Vector_Database_MendozaGarciaArtemio_code.ipynb```](/src/Vector_Database_MendozaGarciaArtemio_code.ipynb) contains  all the project code, including explanations.

### 5.1. Experimental Setup

The key components of our experimental setup included:

**Data Preparation:** Images from the CELEBA dataset were preprocessed to normalize the lighting and alignment to ensure consistency in the data input to the facial recognition system.

**Embedding Extraction:** Using the DeepFace library, facial embeddings were extracted from each image. These embeddings, which are high-dimensional numerical representations of the facial features, were then stored in the Milvus vector database.

**Milvus Database:** Configured to optimize retrieval times and accuracy, the database was crucial in handling the vast amount of embeddings, facilitating efficient storage and retrieval operations.

### 5.2. Use Cases

The experiments were structured around two primary use cases to demonstrate the system's capabilities in different scenarios:

#### 5.2.1. Single Person Matching

Objective: To test the system’s ability to match a single person's image against a database of celebrity embeddings.

Procedure: A single facial image captured via the webcam was processed using DeepFace to extract embeddings. These embeddings were then queried against the stored embeddings in the Milvus database to find the closest match.

![Use Case 1](https://raw.githubusercontent.com/csci-e-104/csci-104-FinalProject/main/images/use-case1-capture.png)

Results: The system was able to identify and retrieve the most similar celebrity images from the database, demonstrating its effectiveness in handling individual facial recognition tasks.

![Use Case 1](https://raw.githubusercontent.com/csci-e-104/csci-104-FinalProject/main/images/use-case1.png)

Results using image similarity.

#### 5.2.2. Group Matching

Objective: To evaluate the system's ability to handle multiple faces in a single image, identifying each person and matching them against the database.

Procedure: An image containing multiple faces was input into the system. DeepFace was used to detect each face within the group and extract their embeddings separately. Each set of embeddings was then queried against the database to find the closest matches for each individual.

![Results Use Case 1](https://raw.githubusercontent.com/csci-e-104/csci-104-FinalProject/main/images/use-case2.png)

DeepFace  discerned the different faces in the group picture. We picked image 1 to run a match.

Results: This test showcased the system’s capability to accurately process and match multiple faces simultaneously. It successfully identified each face in the group and found their closest matches in the database, highlighting the system's scalability and robustness in more complex scenarios.

![Results Use Case 1](https://raw.githubusercontent.com/csci-e-104/csci-104-FinalProject/main/images/use-case2-second-picture.png)

Matched face against our Celebrity DB. The match is perfect.

### 5.3. Evaluation Metrics
To assess the effectiveness and robustness of the facial recognition system, the following evaluation metrics and visualizations were employed:

#### 5.3.1. Barcode Visualization of Embeddings

Methodology: The embeddings extracted from the images were transformed into a two-dimensional matrix by replicating the embedding vectors multiple times. This transformation resulted in a 'barcode' style visualization where each embedding could be visually represented as a series of bars.

![Results Use Case 1](https://raw.githubusercontent.com/csci-e-104/csci-104-FinalProject/main/images/results-use-case1_.png)


Purpose: This approach allowed for a straightforward visual comparison between the query image and the closest matches retrieved from the database. By comparing the barcode patterns, users could easily discern the similarity between faces based on the consistency and variation in the bar patterns.

Application: During experiments, barcode visualizations were used to quickly validate and demonstrate the accuracy of the match results provided by the system. This method proved particularly useful in illustrating visually the embedding process representing facial features.

#### 5.3.2. Dimensionality Reduction and Clustering

PCA (Principal Component Analysis): PCA was applied to the facial embeddings to reduce the dimensionality of the data to two and three dimensions. This reduction helped in visualizing the distribution and grouping of the embeddings in a lower-dimensional space.

K-means Clustering: After reducing the dimensions of the embeddings, K-means clustering was utilized to categorize the embeddings into distinct clusters based on their similarity.

![Results Use Case 1](https://raw.githubusercontent.com/csci-e-104/csci-104-FinalProject/main/images/results-use-case2.png)

Purpose: The combination of PCA and K-means provided a quantitative and visual method to evaluate the similarity among embeddings. By observing how embeddings cluster together in reduced dimensional spaces, we could infer the effectiveness of the facial recognition system in grouping similar faces and distinguishing between different ones.

Application: This method was essential in validating the system’s ability to recognize and categorize multiple faces accurately. It was especially useful in the group matching use case, where the system had to identify and match multiple individuals simultaneously.

## 6. Analysis

### 6.1. What Worked

The integration of the Milvus with the DeepFace library demonstrated significant advantages in the system's ability to accurately match individual and multiple faces against a vast database, highlighting its potential utility in various real-world applications. 

Visual methods like barcode plots and dimensionality reduction through PCA combined with K-means clustering successfully illustrated the similarities between facial embeddings, providing an intuitive way to validate the effectiveness of the facial recognition process.

### 6.2. Challenges Encountered

Despite the successes, several challenges were faced, particularly in handling the sheer volume of data within the CELEBA dataset

#### Long loading time

The entire CelebA dataset was processed in about 10 hours, which is a long time considering that in real world scenarios, the datasets can be higher in order of magnitud that the 200,000 images in our dataset.

#### Parallel Processing Attempts: 

In an effort to manage the large dataset more efficiently, parallel processing techniques were implemented to expedite the task of extracting embeddings and inserting them into the Milvus database. Two approaches were tested:

* __Concurrent Futures__: Utilizing the ```concurrent.futures``` module for threading aimed to parallelize the embedding extraction process. However, this approach did not succeed as anticipated; the Python kernel frequently crashed, which halted progress and required repeated restarts of the process.

* __Dask for Distributed Computing__: As an alternative, the Dask library, known for its capability in handling distributed computing tasks, was employed to achieve parallelism across multiple cores and even different machines. Similar to the threading approach, this also led to kernel crashes, disrupting the workflow and preventing the successful completion of the task.

#### Kernel Instability: 

The repeated kernel crashes during attempts at parallel processing posed significant setbacks. This instability highlights a critical limitation in the current setup, suggesting that either the computational resources were insufficient, or there were underlying issues with how the parallel processes were managed within the Python environment.

### 6.3. Lessons Learned

Complexity of Parallel Processing: The difficulties encountered with concurrent.futures and Dask underline the complexity of implementing parallel processing in Python, especially when interfacing with large datasets and external databases. The kernel crashes indicate that more robust error handling and resource management strategies are required.

Need for Further Research: Determining the precise cause of the kernel instability necessitates further investigation. This could involve examining the memory management during the execution of parallel tasks, the threading model used by Python, or potential conflicts between the parallel processing libraries and the Python interface with Milvus.

Optimization Strategies: Future projects could benefit from exploring different parallel processing frameworks or possibly integrating more stable and scalable technologies such as Apache Spark or using containers to isolate and manage resources more effectively.

Advanced Architecture: The project could probably benefit from a Cluster installation of Milvus instead of a Standalone instance.

## 7. Conclusion

### 7.1. Summary

This project successfully demonstrated the application of vector databases for facial recognition using the CELEBA dataset. By employing the DeepFace library to extract embeddings and Milvus as the vector database for storage and retrieval, we were able to efficiently identify and compare facial similarities among different images. The project showcases the integration of advanced deep learning techniques with vector database technology, offering a scalable solution for image-based data retrieval systems.

### 7.2. Achievements
The use of the Milvus vector database provided a robust framework for handling large datasets of facial embeddings. The system proved capable of performing high-speed searches and maintaining high accuracy in similarity detection, which are critical features for real-world applications in security, marketing, and entertainment industries.

### 7.3. Challenges and Resolutions
Setting up the Milvus vector database presented initial challenges, particularly in configuring the system to efficiently handle the large volume of data. However, these challenges were overcome by following detailed documentation and adjusting system parameters to optimize performance. Additionally, processing the large CELEBA dataset required significant computational resources, highlighting the need for robust hardware when handling big data.

### 7.4. Lessons Learned
The project highlighted the importance of precise data preprocessing and the effectiveness of vector databases in managing large sets of complex data. It also emphasized the necessity for continuous learning and adaptation in the field of deep learning, as the implementation of such systems requires an understanding of both software and hardware capabilities.

### 7.5. Future Work
Looking forward, there are several avenues for further enhancement:

Scalability: Testing the system with even larger datasets and across multiple nodes could provide insights into further scalability options.

Accuracy Improvement: Incorporating more sophisticated facial recognition algorithms could improve the accuracy of the embeddings.

Real-time Processing: Developing capabilities for real-time data processing and retrieval could broaden the application scope to include live video analysis.

Diverse Datasets: Expanding the system to include diverse datasets could help in reducing biases and improving the robustness of the facial recognition technology.


In conclusion, the project not only validated the efficacy of using vector databases for managing large-scale image data but also provided a solid foundation for future research and development in this exciting area of technology.