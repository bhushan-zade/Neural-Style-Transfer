<img src=https://img.shields.io/badge/build%20with-python-yellow><img src="https://img.shields.io/badge/-streamlit-orange"> <img src="https://img.shields.io/badge/deployed%20in-Streamlit Cloudu-blue"> <img src="https://img.shields.io/badge/domain-Deep%20Learning-orange%20.svg" ><img src="https://img.shields.io/badge/Computer%20Vision-orange.svg"><img src="https://img.shields.io/badge/%20Transfer%20Learning%20-%20Pretrained%20Model-orange%20.svg"><img src="https://img.shields.io/badge/Tensorflow%20hub-orange.svg">


# Neural Style Transfer 

Neural Style Transfer (NST) is an optimization technique that involves the utilization of deep convolutional neural network and algorithms to extract the content information from an image (content image) and the style information from another reference image (such as an artwork by a famous painter)—and blend them together so the output image looks like the content image, but “painted” in the style of the style reference image.
 
This technique is used by many popular android iOS apps such as Prisma, DreamScope, PicsArt.

# Tensorflow Hub

TensorFlow Hub is a collection of trained machine learning models that you can use with ease. TensorFlow’s official description for the Hub is as follows:

    TensorFlow Hub is a repository of trained machine learning models ready for fine-tuning and deployable anywhere.
    Reuse trained models like BERT and Faster R-CNN with just a few lines of code.  
    
Apart from pre-trained models such as BERT or Faster R-CNN, there are a good amount of pre-trained models. The one **I used for Project** is **Magenta’s Arbitrary Image Stylization network**. Let’s take a look at what Magenta is.

## Magenta and Arbitrary Image Stylization

**`What is Magenta?`**

Magenta is an open-source research project, backed by Google, which aims to provide machine learning solutions to musicians and artists. Magenta has support in both Python and Javascript. Using Magenta, we can create songs, paintings, sounds, and more. For this project, I used a neural network which is trained and maintained by the Magenta team for Arbitrary Image Stylization.

## Arbitrary Image Stylization

After observing that the original work for NST proposes a slow optimization for style transfer, the Magenta team developed a fast artistic style transfer method, which can work in real-time. Even though the customizability of the model is limited, it is satisfactory enough to perform a non-photorealistic rendering work with NST. Arbitrary Image Stylization under TensorFlow Hub is a module that can perform fast artistic style transfer that may work on arbitrary painting styles.


## Working with Example

**`Get the image from user`**

We will start by selecting two image files. We can directly load these image files with given options.

    1.Upload (Upload Image from System i.e. from Mobile, Tab, Laptop, Computer) 
    2.Camera (Take Picture from System's Camera) 
    3.URL (Provide URL Link of Image)

**`Content Image`**

<img align="centre" alt="coding" height="200" width="220" src="https://user-images.githubusercontent.com/118050962/217671249-454b25c6-3eaf-4e26-b0a9-629ff8477a45.png">

We are free to choose any photo we want. The content image I selected for this project is my current profile photo of github profile, as you can see in Image.

**`Style/Painting Image`**

<img align="centre" alt="coding" height="200" width="220" src="https://user-images.githubusercontent.com/118050962/217672014-77547edc-a366-4bff-be23-506b20ae4817.jpg">

I would like to transfer the style of Leonardo da Vinci. So, I chose one of his famous paintings: Mona Lisa, Considered an archetypal masterpiece of the Italian Renaissance, it has been described as "the best known, the most visited, the most written about, the most sung about, the most parodied work of art in the world. You can even use your own drawings.

After Selecting the image we need to do some Image Scaling and processing. and after that we need to load this images in content image and style image. after that We need to load/import Load Magenta's Arbitrary Image Stylization network from TensorFlow Hub. and then we need to predict final stylized image with given input images and with the help of Arbitrary Image Stylization (pre-trained model).

**`Final-> Stylized image`**

<img align="centre" alt="coding" height="200" width="220" src="https://user-images.githubusercontent.com/118050962/217676182-0609260f-ea80-4508-be11-6e6636025736.png">





    
 
