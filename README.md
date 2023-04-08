# 🐕 Dog breed classification 

[Link to the main classification notebook on nbviewer.](https://nbviewer.org/github/ana-bernal/P6_img-classif/blob/main/02_classification.ipynb).

As a first exercice with convolutionnal neural networks I built some models from scratch and also used transfer learning from architectures [VGG16](https://keras.io/api/applications/vgg/#vgg16-function) ([paper](https://arxiv.org/abs/1409.1556)) and [EfficientNetV2S](https://keras.io/api/applications/efficientnet_v2/#efficientnetv2s-function) ([paper](https://arxiv.org/abs/2104.00298)).

A Hugging Face space for classification using the best performing model can be found [here](https://huggingface.co/spaces/ana-bernal/DogBreedClassification).

---

#  🐕 Dog breed recognition: classification

|   |   |
|---|---|
| Project  |    [Classez des images à l'aide d'algorithmes de Deep Learning](https://openclassrooms.com/fr/paths/148/projects/634/assignment)         |
| Date   |   April 2023   |
| Autor  | Ana Bernal                                                    |
| Data source | [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) |
| Mentor | Samir Tanfous | 
| Number of notebooks  | 2                                                   |

**Description:** The aim of this project is to develop an algorithm that classifies pictures of dogs in breeds.

We explore two different approaches for this task: We create convolutionnal neural networks (CNN) from scratch and we use **transfer learning**.

The best approach turns out to be transfer learning.

All along the classification notebook there are notes documenting the process and the final choice. At the end we train the best model on all the data and export it to a Hugging Face hub. It is ready to use in a online API:

[[click here to go to API space in HuggingFace]](https://huggingface.co/spaces/ana-bernal/DogBreedClassification)
