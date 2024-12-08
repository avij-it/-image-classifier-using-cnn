# Cats vs Dogs Image Classifier

## Project Overview

This is a Convolutional Neural Network (CNN) image classification project that can distinguish between images of cats and dogs using TensorFlow and TFLearn. The model is trained on a dataset of cat and dog images and uses a deep learning approach to classify new images.

## Prerequisites

- Python 3.x
- OpenCV (cv2)
- NumPy
- TensorFlow
- TFLearn
- Matplotlib
- tqdm

## Installation

1. Clone the repository
2. Install required dependencies:
```bash
pip install opencv-python numpy tensorflow tflearn matplotlib tqdm
```

## Dataset Structure

```
dataset/
├── Cats_vs_Dogs/
│   ├── train/
│   │   ├── cat.xxx.jpg
│   │   ├── dog.xxx.jpg
│   └── test1/
│       ├── x.jpg
│       └── y.jpg
```

## Project Components

### Data Preprocessing
- `label_img()`: Assigns binary labels to images (Cat: [1,0], Dog: [0,1])
- `create_train_data()`: Loads and preprocesses training images
- `process_test_data()`: Loads and preprocesses test images

### Neural Network Architecture
- Input layer: 50x50 grayscale images
- 5 Convolutional layers with ReLU activation
- Max pooling after each convolutional layer
- Fully connected layer with 1024 neurons
- Dropout layer (0.8 rate)
- Softmax output layer for classification

### Training
- Optimizer: Adam
- Learning Rate: 0.001
- Loss Function: Categorical Cross-Entropy
- Epochs: 5

## Usage

1. Prepare your dataset in the specified directory structure
2. Run the script to:
   - Preprocess images
   - Train the model
   - Save the model
   - Visualize predictions on test images

## Model Performance
The model will output training metrics during the fitting process, including:
- Validation accuracy
- Loss progression

## Visualization
- Generates a matplotlib figure showing the first 20 test images
- Displays predicted labels (Cat/Dog) on each image

## Saved Outputs
- `train_data.npy`: Saved preprocessed training data
- `test_data.npy`: Saved preprocessed test data
- `dogsvscats-{learning_rate}-{model_description}.model`: Saved trained model
- `log/`: TensorBoard logs for model training analysis

## Customization
- Adjust `IMG_SIZE` to change image dimensions
- Modify learning rate (`LR`)
- Experiment with network architecture
- Change dropout rate

## Limitations
- Requires a balanced dataset
- Performance depends on image quality and dataset diversity
- Currently supports grayscale images

## Future Improvements
- Add data augmentation
- Implement transfer learning
- Support color image processing
- Add more robust error handling
