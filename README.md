# A Gripping Story - Written by Deep Neural Network

### Team members

[Deepan Chakravarthi Padmanabhan](https://github.com/DeepanChakravarthiPadmanabhan)

[Venkata Santosh Sai Ramireddy Muthireddy](https://github.com/santoshreddy254/)

Vahid MohammadiGahrooei


### Pipeline

![Pipeline](https://user-images.githubusercontent.com/43172178/86264098-6160ba00-bbc2-11ea-84c0-d6cd8999f99c.png)

<p align="center">
  Figure 1: Two-stage pipeline of Neural-storyteller.
</p>

### Usage

#### Image Captioning module

1. Navigate to the image captioning module.

2. To download dataset:

```python download_data.py```

3. To train the model:

```python main.py --architecture Inception --optimizer Adam --num_epochs 20 --num_examples 10000 --annotation_folder annotations/ --image_folder train2014```

4. To predict captions given image:

```python evaluate.py --test_image <test_image_file_path>```

For detailed description on available arguments for main.py and evaluate.py run:

```python main.py --help```
#### Story generation module
1. Check python notebook for story generation [notebook](https://github.com/DeepanChakravarthiPadmanabhan/Image_Storyteller/blob/master/Stroy_Generator.ipynb)
### Documents

Kindly refer the presentation of the project work [here](https://github.com/DeepanChakravarthiPadmanabhan/Image_Storyteller/blob/master/Documents/Presentation_AGrippingStory-WrittenbyDeepNeuralNetwork.pdf).

### Declaration

This is a project work done for Natural Language Processing - Summer Semester 2020 course work at Hochschule Bonn-Rhein-Sieg.

### Acknowledgement

The code is adapted from TensorFlow - Image captioning tutorial for image captioning.

### Reference

1. Yukun Zhu et al. “Aligning Books and Movies: Towards Story-like Visual Explanations by Watching Movies and Reading Books”. In: arXiv preprint arXiv:1506.06724 (2015).
2. MD Zakir Hossain et al. “A comprehensive survey of deep learning for image captioning”. In: ACM Computing Surveys (CSUR) 51.6 (2019), pp. 1–36.
3. Tsung-Yi Lin et al. “Microsoft coco: Common objects in context”. In: European conference on computer vision. Springer. 2014, pp. 740–755.
4. Kelvin Xu et al. Show, Attend and Tell: Neural Image Caption Generation with Visual Attention. 2015. arXiv: 1502.03044 [cs.LG].
5. Tensorflow. Tensorflow - Image captioning. Accessed on: 20-06-2020. [Online]. 2018. url: https://www.tensorflow.org/tutorials/text/image_captioning.
6. Tensorflow. Tensorflow - Text generation. Accessed on: 20-06-2020. [Online]. 2018. url: https://www.tensorflow.org/tutorials/text/text_generation.
7. Sanyam Agarwal. My thoughts in Skip-thoughts. Accessed on: 20-06-2020. [Online]. 2018. url:
https://medium.com/@sanyamagarwal/my-thoughts-on-skip-thoughts-a3e773605efa.
8. Ryan Kiros et al. “Skip-Thought Vectors”. In: arXiv preprint arXiv:1506.06726 (2015).
9. Gabriel Loye. Attention mechanisms. Accessed on: 20-06-2020. [Online]. 2019. url: https://blog.floydhub.com/attention- mechanism.




