#editing for lasting
import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd
import fitz
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from PIL import Image
import tempfile
from utils import ProfileMatcher  # Our core matching logic

# App configuration
st.set_page_config(
    page_title="Aluminum Profile Matcher",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("assets/styles.css")

# Initialize session state
if 'matches' not in st.session_state:
    st.session_state.matches = None
if 'processed_img' not in st.session_state:
    st.session_state.processed_img = None
if 'user_img' not in st.session_state:
    st.session_state.user_img = None

# Sidebar for configuration
with st.sidebar:
    st.title("Configuration")
    template_root = st.text_input("training_data", "Profile_Templates")
    excel_path = st.text_input("data.xlsx", "Profile_Specifications.xlsx")
    pdf_folder = st.text_input("data", "data")
    
    # Initialize matcher
    matcher = ProfileMatcher(template_root, excel_path, pdf_folder)

# Main app
st.title("Aluminum Profile Matcher")
st.markdown("Upload an image or draw a profile to find matching aluminum sections")

# Input method selection
input_method = st.radio(
    "Select input method:",
    ["Upload Image", "Draw Profile"],
    horizontal=True
)

if input_method == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload a profile image", 
        type=["png", "jpg", "jpeg"]
    )
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        user_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        st.session_state.user_img = user_img
        
        with st.spinner("Finding matching profiles..."):
            processed_user, matches = matcher.find_similar_profiles(user_img)
            st.session_state.matches = matches
            st.session_state.processed_img = processed_user
            st.success("Matching completed!")

else:  # Draw Profile
    st.markdown("Draw your profile sketch below:")
    
    # Create drawing canvas
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=3,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=400,
        width=400,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Drawing"):
            st.session_state.canvas_key += 1  # Reset canvas
    with col2:
        if st.button("Search Drawing"):
            if canvas_result.image_data is not None:
                # Convert canvas to grayscale image
                img_array = np.array(canvas_result.image_data)
                gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
                gray_img = 255 - gray_img  # Invert colors
                
                with st.spinner("Finding matching profiles..."):
                    processed_user, matches = matcher.find_similar_profiles(gray_img)
                    st.session_state.matches = matches
                    st.session_state.processed_img = processed_user
                    st.session_state.user_img = gray_img
                    st.success("Matching completed!")

# Display results if available
if st.session_state.matches is not None:
    st.header("Matching Results")
    
    # Show input images
    col1, col2 = st.columns(2)
    with col1:
        st.image(
            st.session_state.user_img, 
            caption="Original Input", 
            use_column_width=True
        )
    with col2:
        st.image(
            st.session_state.processed_img, 
            caption="Normalized Input", 
            use_column_width=True
        )
    
    # Show matches
    st.subheader("Top Matches")
    cols = st.columns(len(st.session_state.matches))
    for idx, match in enumerate(st.session_state.matches):
        with cols[idx]:
            st.image(
                match['processed'], 
                caption=f"Match {idx+1}: {match['class']} (Score: {match['similarity']:.3f})",
                use_column_width=True
            )
    
    # Create selection interface
    selected_class = st.selectbox(
        "Select a profile to view details:",
        [match['class'] for match in st.session_state.matches]
    )
    
    if selected_class:
        # Get specifications
        row = matcher.find_matching_row(selected_class)
        pdf_image = matcher.get_pdf_as_image(selected_class)
        
        # Display specifications
        st.subheader(f"Specifications for: {selected_class}")
        if row is not None:
            spec_df = pd.DataFrame.from_dict(
                {k: [v] for k, v in row.items() if k != 'Class' and pd.notna(v)},
                orient='index',
                columns=['Value']
            )
            st.dataframe(spec_df)
        else:
            st.warning("No specifications found for this profile")
        
        # Display PDF preview
        if pdf_image is not None:
            st.subheader("Related Document Preview")
            st.image(
                pdf_image, 
                caption=f"PDF for {selected_class}",
                use_column_width=True
            )
        else:
            st.warning("No PDF document found for this profile")

