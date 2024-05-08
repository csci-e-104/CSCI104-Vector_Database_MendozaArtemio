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

It is a stablished fact that aliens walk among us, as demonstrated by the documental MIB (**Men In Black, Will Smith et all. 1997**). However, little we know if a celebrity is disguised as a regular person, living in our same neighborour, goes to the same grocery store than we go, or, if she is taking the same course, CS104-Advanced deep learning, and we are not aware?

We will use a Vector Database, with more than two hundred thousand vector embeddings of more than ten thousand celebrities, to try to shed some light on this potential problematic situation.

### But, from where do we get Celebrities images?

We start with the [Dataset CELEBA, a well known collection of images from the university of Hong Kong](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). It contains more than 200,000 images from more than 10,000 celebrities. 

For this project, we use the ["in-the-wild" images, which are availble for download as a compressed file in this google drive](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg&usp=sharing).  

The compressed file size exceeds the 1.3 GB mark, and took about one hour to decompress. It was stored locally to speed the insert process.

### And, how do we extract the Vector Embeddings?

We use DeepFace, which abstract embedding extraction using different CNN SOTA models:

* FaceNet
* VGG_Face (University of Oxford)
* OpenFace
* DeepFace

For this project, we used FaceNet. FaceNet was developed by Google in 2015. With 140 million of parameters and 22-layer depth, it achieves a prediction accuracy of 99.22% on LFW dataset. With this model Google introduced the *triplet loss function*, which works using forming triplets, with one anchor, positive example, and negative example. 

The image above is a high level schema of the FaceNet CNN architecture

![Face Neet Architecture](https://raw.githubusercontent.com/csci-e-104/csci-104-FinalProject/main/images/deep-learning-architecture.png)


### What about where to store the Vector Embeddings?
For this project we are used Milvus, an Open-source Vector Database

Milvus can be installed locally locally or in the cloud. For this project, we installed the Milvus locally, as a standalone instance with GPU support, using the Docker image provided in the documentation manual.

The standalone instance includes three components:

Milvus: the core functional component.
Meta Store: the medata engine, which access and stores metadata of Milvus'internal components, including proxies, index nodes, and more.
Object Storage: The storage engine, which is responsible for data persistence for Milvus


![Face Neet Architecture](https://raw.githubusercontent.com/csci-e-104/csci-104-FinalProject/main/images/milvus_standalone_architecture.jpg)


The image above is a high level schema of the Milvus Standalone architecture

Now that we stablished the real motivations behind this project, lets get to the serious, boring documentation.

But rest assure, we'll get back in track by the end of it to get an answer to our imposed question.

---

### Abstract 
[Abstract](./docs/Vector_Database_MendozaGarciaArtemio_onePage.md)

### Final Report
[Final Report](./docs/final_report.md)