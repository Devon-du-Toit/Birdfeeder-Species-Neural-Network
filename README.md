# Birdfeeder Species Neural Network

This repository contains a series of TensorFlow experiments for identifying animals visiting a bird feeder. The project evolves from a simple binary classifier to a six-class model, adding new species at each stage.

## Model Progression and Code Structure

- **Binary-Squirrel-Sparrow-Classifier** – Initial experiment distinguishing Cape Sparrow and Squirrel. `binary_squirrel_sparrow_trainer.ipynb` loads images with `image_dataset_from_directory`, builds a four-layer convolutional network, trains with early stopping, and saves the model. `squirrel_or_sparrow_impl.ipynb` demonstrates loading the trained network and classifying individual images.
- **3-Species-Identifier** – Adds Masked Weaver as a third class. `trainer_3_species.ipynb` augments training data with random flips, rotations, zoom, and contrast adjustments, then trains and evaluates a softmax CNN; `model_architecture.png` visualizes the model.
- **5-Species-Identifier** – Introduces Cape White-eye and Swee Waxbill. `trainer_5_species.ipynb` follows the augmented pipeline, uses an early-stopping callback to prevent overfitting, and saves a five-class model.
- **6-Species-Identifier** – Final stage adding Southern Double-collared Sunbird. `trainer_6_species.ipynb` trains on all six species with augmentation and early stopping, logs validation accuracy, and exports the final network.

Supplementary notebooks at the repository root support deployment workflows:

- **model_impl.ipynb** – Loads a trained model and runs predictions on sample images.
- **model_LITE_converter.ipynb** – Converts saved Keras models to TensorFlow Lite using `tf.lite.TFLiteConverter` for edge devices.

## Usage

Each training notebook expects a dataset organized with one subfolder per class and uses Keras utilities to load, augment, and batch the images. Execute the notebooks in order to train progressively more capable models or adapt them for your own species list.

## Training workflow

A typical training notebook follows a consistent pattern. For example, the five-species trainer loads images from a directory, splits them into training and validation sets, and records the class labels—including bird species and squirrels—before model training begins.

Data augmentation and normalization steps are defined with Keras’ preprocessing layers to improve model robustness.

Model architectures are straightforward convolutional networks built with a stack of `Conv2D` and pooling layers followed by dense layers for classification.

## Inference and deployment

The `model_impl.ipynb` notebook demonstrates how to load a trained model, preprocess an input image (padding or resizing to 256×256), and run prediction to display the most probable class and confidence score.

For edge deployment, `model_LITE_converter.ipynb` shows how to convert a saved TensorFlow model into a TensorFlow Lite file using the `TFLiteConverter`, enabling optimizations and avoiding legacy TF1 conversion paths.

## Making use of this repository

1. **Understand the dataset layout**: The notebooks expect an image directory structured by class names (e.g., `../dataset/Cape Sparrow/`). Ensure your dataset matches this layout before training.
2. **Run through a training notebook**: Start with `5-Species-Identifier/trainer_5_species.ipynb` to see the full workflow—loading data, augmenting images, defining the CNN, and training.
3. **Experiment with model architecture**: Try changing layer counts or hyperparameters in the sequential model to improve accuracy or reduce size.
4. **Practice deployment**: Use `model_impl.ipynb` to test inference on new images, then follow `model_LITE_converter.ipynb` to create a `.tflite` model for mobile or embedded devices.
5. **Extend to more species or tasks**: Once comfortable, adapt the notebooks to new species or create additional binary classifiers for other animals.
