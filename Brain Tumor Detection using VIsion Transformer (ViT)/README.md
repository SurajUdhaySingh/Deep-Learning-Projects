# ViT Brain Tumor Classification

This project uses a Vision Transformer (ViT) model to classify brain tumor types in MRI images. The model is fine-tuned on a custom dataset consisting of four tumor categories: Glioma, Meningioma, Pituitary, and No Tumor. The ViT was introduced in the paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," which allows transformers to be applied directly to images.

In summary, a ViT "sees" the entire image, unlike CNNS whose filter maps are applied locally and exhibit translational invariance. This means that ViTs are able to capture global dependencies, and if trained on sufficiently large dataset, they surpass CNNs of near-equal number of parameters.

Their working can be explained as follows, the input images are uniformly split into "patches," where each patch size is of typically 16x16 dimensions, relative to the image. The resultant patches are then embedded with a learnable positional encoding, and then the patches are linearly projected, with the projection being learnable as well. The rest is application of multi-head attention, after which a classification head is applied, which outputs logits for each class.

# Project and Directory Structure
"test_images" contains sample MRI scans for testing, taken from the dataset (NOT FROM THE TRAINING SPLIT.)
"vit-brain-mri-tumor" contains the checkpoints.
"vit-braintumor-mri" contains the fine-tuned ViT's parameters.
"train-model.ipynb" contains the preprocessing and training code.
"inference.ipynb" contains some manual testing.

NOTE: I HAD TO THE CHECKPOINTS AND MODEL WEIGHTS, BECAUSE GITHUB WOULDN'T LET ME POST THEM.

# Requirements

To run this project, you'll need to install the following dependencies:

- Python 3.7+
- PyTorch
- Hugging Face `transformers` library
- Datasets library
- matplotlib
- PIL (Pillow)

