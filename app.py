import streamlit as st 
from PIL import Image
from rembg import remove 

def main():
    st.title("Alpha Matting")
    uploaded_files = st.file_uploader("Upload an image",type=['png','jpg','jpeg'],accept_multiple_files=False)
    if st.button("Generate"):
        if uploaded_files:
            img = Image.open(uploaded_files)
            res = bg_remover(img)
            st.balloons()
            col1,col2 = st.columns(2,gap="large")
            with col1:
                st.write("### Original Image")
                st.image(img)
            with col2:
                st.write("### Result Image")
                st.image(res) 
        else:
            st.error("Please upload a valid image")

def bg_remover(image):
    return remove(image)


if __name__ == "__main__":
    main()