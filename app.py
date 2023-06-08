import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from transformers import pipeline

# Models used for caption generation
transformer_models = {
    "Model 1": "Salesforce/blip-image-captioning-base", # https://huggingface.co/Salesforce/blip-image-captioning-base
    "Model 2": "Salesforce/blip-image-captioning-large", # https://huggingface.co/Salesforce/blip-image-captioning-large
    "Model 3": "ydshieh/vit-gpt2-coco-en" # https://huggingface.co/ydshieh/vit-gpt2-coco-en
}

# Function to generate captions
def generate_captions(image, model_name):
    caption_generator = pipeline('image-to-text', model=model_name)
    captions = caption_generator(image)
    return captions


def main():
    st.title("IMGWhisper: An Image Caption Generator")

    image_source = st.radio("Choose image source:", ("Upload an image", "Provide an image URL"))

    # Load the image
    if image_source == "Upload an image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
    else:
        image_url = st.text_input("Enter the image URL")
        if image_url:
            try:
                response = requests.get(image_url)
                image = Image.open(BytesIO(response.content))
            except:
                st.error("Error: Failed to load image from the provided URL")

    # Display the image
    if "image" in locals():
        st.image(image, caption="Uploaded/Provided Image", width=300)

        num_captions = st.slider("How many captions do you want to generate?", min_value=1, max_value=3, value=1)

        if st.button("Generate Caption"):
            captions = []
            for i in range(num_captions):
                model_name = transformer_models[f"Model {i+1}"]
                caption = generate_captions(image, model_name)
                captions.append(caption[0]['generated_text'])

            # Display the captions
            st.header("Generated Captions:")
            for i, caption in enumerate(captions):
                st.subheader(f"Caption {i+1}: {caption}")

# Run the app
if __name__ == "__main__":
    main()