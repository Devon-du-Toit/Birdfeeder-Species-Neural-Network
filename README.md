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
