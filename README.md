# Computer Vision Projects

This repository encompasses a collection of seven distinct computer vision projects, each designed to address specific challenges and applications within the field. The projects demonstrate various techniques and methodologies in computer vision, utilizing a range of technologies and frameworks.

## Tech Stack

The projects in this repository leverage the following technologies and libraries:

- **Programming Languages:** Python
- **Libraries and Frameworks:**
  - OpenCV
  - TensorFlow
  - Keras
  - NumPy
  - Matplotlib

## Installation Instructions

To set up the repository locally:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/fahad-git/computer-vision-projects.git
   ```
2. **Navigate to the Repository Directory:**
   ```bash
   cd computer-vision-projects
   ```
3. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   ```
4. **Activate the Virtual Environment:**
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On Unix or MacOS:
     ```bash
     source venv/bin/activate
     ```
5. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

*Note:* Each project may have additional dependencies or setup instructions detailed in their respective sections below.

## Contributing Guidelines

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

Please ensure your code adheres to the project's coding standards and includes appropriate tests.

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

Below are the individual projects contained in this repository:

## 1. Autoencoder

**Purpose:**
Develop an autoencoder model for unsupervised feature learning and dimensionality reduction.

**Tech Stack:**
- Python
- TensorFlow
- Keras

**Installation Instructions:**
Navigate to the `Autoencoder` directory and ensure the required libraries are installed as per the main installation instructions.

**Usage Guide:**
Run the `autoencoder.py` script to train the model on the dataset.

**Features:**
- Encoder and decoder architecture
- Reconstruction of input data

**Dependencies:**
- TensorFlow
- Keras

## 2. Autoencoder Fashion MNIST Dataset Model

**Purpose:**
Implement an autoencoder specifically trained on the Fashion MNIST dataset for feature extraction.

**Tech Stack:**
- Python
- TensorFlow
- Keras

**Installation Instructions:**
Navigate to the `Autoencoder_fashion_mnist_dataset_model` directory.

**Usage Guide:**
Execute the `fashion_mnist_autoencoder.py` script to train and evaluate the model.

**Features:**
- Training on Fashion MNIST dataset
- Visualization of reconstructed images

**Dependencies:**
- TensorFlow
- Keras

## 3. Coil 20 Unprocessed

**Purpose:**
Explore object recognition using the unprocessed COIL-20 dataset.

**Tech Stack:**
- Python
- OpenCV
- NumPy

**Installation Instructions:**
Navigate to the `Coil_20_Unproc` directory.

**Usage Guide:**
Use the `coil20_processing.py` script to preprocess and analyze the dataset.

**Features:**
- Data preprocessing
- Feature extraction

**Dependencies:**
- OpenCV
- NumPy

## 4. Dog Cat Classification

**Purpose:**
Classify images of dogs and cats using a convolutional neural network.

**Tech Stack:**
- Python
- TensorFlow
- Keras

**Installation Instructions:**
Navigate to the `Dog_Cat_Classification` directory.

**Usage Guide:**
Run the `dog_cat_classifier.py` script to train and test the model.

**Features:**
- Image classification
- Data augmentation

**Dependencies:**
- TensorFlow 

## 5. **Caltech 101 Dataset Model**

**Purpose:**  
Create and evaluate a model trained on the Caltech 101 dataset to classify objects into various categories.

**Tech Stack:**  
- Python  
- TensorFlow  
- Keras  

**Installation Instructions:**  
1. Navigate to the `caltech_101_dataset_Model` directory.  
2. Ensure the Caltech 101 dataset is available or downloaded.  
3. Follow the repository's general installation instructions for dependencies.  

**Usage Guide:**  
- Run the `caltech_101_model.py` script to train and evaluate the model.  
- Ensure the dataset is correctly structured before execution.  

**Features:**  
- Multi-class classification on Caltech 101 dataset.  
- Support for training visualization using Matplotlib.  

**Dependencies:**  
- TensorFlow  
- Keras  

---

## 6. **Flower Recognition**

**Purpose:**  
Recognize and classify different types of flowers using image classification techniques.  

**Tech Stack:**  
- Python  
- TensorFlow  
- Keras  

**Installation Instructions:**  
1. Navigate to the `flower_recognition` directory.  
2. Install the required dependencies listed in the `requirements.txt` file.  

**Usage Guide:**  
- Execute the `flower_recognition.py` script to train the model on the flower dataset.  
- Follow any dataset preprocessing steps mentioned in the script.  

**Features:**  
- Classification of flower species.  
- Visualization of training and testing accuracy.  

**Dependencies:**  
- TensorFlow  
- Keras  

---

## 7. **Weather Image Dataset**

**Purpose:**  
Classify weather conditions (e.g., sunny, cloudy, rainy) based on image data.  

**Tech Stack:**  
- Python  
- TensorFlow  
- Keras  

**Installation Instructions:**  
1. Navigate to the `weather_image_dataset` directory.  
2. Ensure the weather dataset is available locally or configured for download.  
3. Follow the general installation steps to set up the environment.  

**Usage Guide:**  
- Run the `weather_classification.py` script to train and test the model.  
- Use the trained model to predict weather conditions on new images.  

**Features:**  
- Classification of images into weather categories.  
- Pretrained model support for transfer learning (if applicable).  

**Dependencies:**  
- TensorFlow  
- Keras  
