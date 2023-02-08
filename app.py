import streamlit as st
import os
import tensorflow_hub as hub
from myfunctions import load_img, transform_img, tensor_to_image, imshow, export_image
import tensorflow as tf




# Only use the below code if have low resources.
os.environ['CUDA_VISIBLE_DEVICES'] = ""
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

# For supressing warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

st.set_page_config(page_title="Neural Style Transfer-Bhushan", layout="wide")
st.write("""
# Neural Style Transfer
""")

# Load Pretrained Model
@st.cache
def load_model():
    model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    return model


model_load_state = st.text('Loading Model')
model = load_model()
# Notify that the data was successfully loaded.
model_load_state.text('')


content_image, style_image, final = st.columns(3)

with content_image:
    st.write('## Content Image')
    chosen_content = st.sidebar.radio(
        '  ',
        ("Upload", "Camera", "URL"),)

# 1.OPTION-> UPLOAD
    if chosen_content == 'Upload':
        st.write(f"You choosed {chosen_content}!")
        content_image_file = st.sidebar.file_uploader(
            "Pick a Content image", type=("png", "jpg", "jpeg"))
        try:
            content_image_file = content_image_file.read()
            content_image_file = transform_img(content_image_file)
        except:
            pass


# 2.OPTION-> CAMERA
    elif chosen_content == 'Camera':
        st.write(f"You choosed {chosen_content}!")
        content_image_file = st.sidebar.camera_input("Take a picture")
        try:
            content_image_file = content_image_file.read()
            content_image_file = transform_img(content_image_file)
        except:
            pass

# 3.OPTION-> URL
    elif chosen_content == 'URL':
        st.write(f"You choosed {chosen_content}!")
        url = st.sidebar.text_input('URL for the content image.')

        try:
            content_path = tf.keras.utils.get_file(os.path.join(os.getcwd(), 'content.jpg'), url)
        except:
            pass

        try:
            content_image_file = load_img(content_path)
        except:
            pass

# SHOW CONTENT IMAGE
    try:
        st.write('Content Image')
        st.image(imshow(content_image_file))
    except:
        pass


# STYLE IMAGE
with style_image:
    st.write('## Style/Painting Image')
    chosen_style = st.sidebar.radio(
        ' ',
        ("Upload", "URL"))

# 1.OPTION-> UPLOAD
    if chosen_style == 'Upload':
        st.write(f"You choosed {chosen_style}!")
        style_image_file = st.sidebar.file_uploader(
            "Pick a Style/Painting image", type=("png", "jpg", "jpeg"))
        try:
            style_image_file = style_image_file.read()
            style_image_file = transform_img(style_image_file)
        except:
            pass

# 2.OPTION-> URL
    elif chosen_style == 'URL':
        st.write(f"You choosed {chosen_style}!")
        url = st.sidebar.text_input('URL for the style image.')
        try:
            style_path = tf.keras.utils.get_file(
                os.path.join(os.getcwd(), 'style.jpg'), url)
        except:
            pass
        try:
            style_image_file = load_img(style_path)

        except:
            pass

# SHOW STYLE IMAGE
    try:
        st.write('Style Image')
        st.image(imshow(style_image_file))
    except:
        pass





with final:
    # FINAL PREDICTION-> NEURAL STYLE TRANSFORMER
    button_style = """
            <style>
            .stButton > button {
                color: black;
                background: white;
                border: 3px solid;
                border-radius: 10px;
                font-size: 150px;
                width: 200px;
                height: 75px;
            }
            </style>
            """
    st.markdown(button_style, unsafe_allow_html=True)

    button_style = """
                <style>
                .stButton > download_button {
                    color: black;
                    background: white;
                    border: 3px solid;
                    border-radius: 10px;
                    font-size: 150px;
                    width: 250px;
                    height: 100px;
                }
                </style>
                """
    st.markdown(button_style, unsafe_allow_html=True)

    predict = st.button("***Start Neural Style Transfer***")
    with st.spinner("Processing..."):
        if predict:
            if content_image_file is not None and style_image_file is not None:
                try:
                    stylized_image = model(tf.constant(content_image_file), tf.constant(style_image_file))[0]
                    final_image = tensor_to_image(stylized_image)
                except:
                    stylized_image = model(tf.constant(tf.convert_to_tensor(content_image_file[:, :, :, :3])),tf.constant(tf.convert_to_tensor(style_image_file[:, :, :, :3])))[0]
                    final_image = tensor_to_image(stylized_image)

                a= st.write('## Final Image')
                st.image(final_image)
                st.download_button(label="**Download Final Image**", data=export_image(stylized_image), file_name="final_image.png",
                                   mime="image/png")

                try:
                    # Delete style.jpg and content.jpg
                    os.remove("style.jpg")
                    os.remove("content.jpg")
                except:
                    pass
            else:
                st.markdown("### Please Upload Both Images")


