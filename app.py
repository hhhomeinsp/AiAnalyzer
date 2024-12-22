import streamlit as st
import openai
from openai import OpenAI
import base64
from PIL import Image
import io
import os
import logging

# --------------------------- #
#       Configuration         #
# --------------------------- #

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize OpenAI client
def init_openai_client():
    """Initialize OpenAI client with API key from Streamlit secrets"""
    try:
        # Try to get API key from Streamlit secrets
        api_key = st.secrets["OPENAI_API_KEY"]
        
        # Initialize client with the API key
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.openai.com/v1"
        )
        
        return client
    except Exception as e:
        st.error("Error initializing OpenAI client. Please check your API key configuration.")
        logging.error(f"OpenAI client initialization error: {e}")
        return None

# Define AI Prompts as Constants
IMAGE_ANALYSIS_PROMPT = (
    "Please analyze the given image along with any provided text context (if any) and provide an analysis "
    "of any deficiencies or conditions, safety concerns, functionality issues, etc."
)

DEFECT_ANALYSIS_PROMPT = (
    "Please analyze the given deficiency comment and provide a more detailed breakdown of the comment "
    "to allow for better understanding."
)

# --------------------------- #
#        AI Functions         #
# --------------------------- #

def process_image(image_file):
    """
    Process and compress uploaded image
    
    Parameters:
        image_file (UploadedFile): The image file uploaded by the user.
        
    Returns:
        str: Base64 encoded image string or error message
    """
    # Define maximum dimensions
    MAX_WIDTH = 800
    MAX_HEIGHT = 800

    try:
        # Open the image
        image = Image.open(image_file)
        
        # Resize the image while maintaining aspect ratio
        image.thumbnail((MAX_WIDTH, MAX_HEIGHT))
        
        # Compress the image by saving it with lower quality
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=70)
        
        # Get the byte data and encode to Base64
        image_bytes = buffered.getvalue()
        return base64.b64encode(image_bytes).decode('utf-8')
        
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        raise Exception(f"Error processing image: {str(e)}")

def analyze_image(image_file, context, prompt, client):
    """
    Analyze image using OpenAI API
    
    Parameters:
        image_file (UploadedFile): The image file uploaded by the user.
        context (str): Additional text context provided by the user.
        prompt (str): The AI prompt guiding the analysis.
        client (OpenAI): OpenAI client instance.
        
    Returns:
        str: The AI-generated analysis or an error message.
    """
    try:
        # Process the image
        base64_image = process_image(image_file)
        
        # Make API request
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",  # Updated to use vision model
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{prompt}\n\nContext: {context}"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content
        
    except Exception as e:
        logging.error(f"Error in analyze_image: {e}")
        return f"Error: {str(e)}"

def analyze_defect(defect_text, prompt, client):
    """
    Analyze defect comment using OpenAI API
    
    Parameters:
        defect_text (str): The defect comment/narrative provided by the user.
        prompt (str): The AI prompt guiding the analysis.
        client (OpenAI): OpenAI client instance.
        
    Returns:
        str: The AI-generated detailed breakdown or an error message.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",  # Updated to use latest GPT-4 model
            messages=[
                {
                    "role": "user",
                    "content": f"{prompt}\n\nDefect Comment: {defect_text}"
                }
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content
        
    except Exception as e:
        logging.error(f"Error in analyze_defect: {e}")
        return f"Error: {str(e)}"

# --------------------------- #
#         Streamlit UI        #
# --------------------------- #

def main():
    """Main function to run the Streamlit app"""
    
    # Set Streamlit page configuration
    st.set_page_config(page_title="Home Inspection AI Assistant", layout="wide")
    st.title("🏠 Home Inspection AI Assistant")

    # Initialize OpenAI client
    client = init_openai_client()
    if client is None:
        st.error("Failed to initialize OpenAI client. Please check your configuration.")
        return

    # Provide informational message
    st.info("📌 Please ensure that your images are clear and within the 5MB size limit for optimal analysis.")

    # Create two tabs: Image Analysis and Defect Information
    tab1, tab2 = st.tabs(["🖼️ Image Analysis", "🔍 Defect Information"])

    # --------------------------- #
    #      Tab 1: Image Analysis  #
    # --------------------------- #
    with tab1:
        st.header("🖼️ Image Analysis")
        
        # File Uploader for Image
        uploaded_file = st.file_uploader("📤 Upload an image (Max 5MB)", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            # Check file size
            file_size = len(uploaded_file.getvalue())
            
            if file_size > 5 * 1024 * 1024:  # 5MB limit
                st.error("❌ The uploaded image exceeds the 5MB size limit. Please upload a smaller image.")
            else:
                try:
                    # Display uploaded image
                    image = Image.open(uploaded_file)
                    st.image(image, caption="📸 Uploaded Image", use_column_width=True)
                except Exception as e:
                    st.error(f"Error displaying image: {str(e)}")
        
        # Additional Context Input
        context = st.text_area(
            "📝 Additional Context (Optional)", 
            "Provide any relevant context about the image (e.g., location, age of home, specific concerns).",
            height=100
        )
        
        if len(context) > 500:
            st.warning("⚠️ Additional context should be under 500 characters.")
            context = context[:500]
        
        # Analyze Button
        if st.button("🔍 Analyze Image", key="analyze_image"):
            if uploaded_file is None:
                st.warning("⚠️ Please upload an image to analyze.")
            else:
                with st.spinner("⏳ Analyzing image..."):
                    analysis = analyze_image(uploaded_file, context, IMAGE_ANALYSIS_PROMPT, client)
                    if analysis:
                        st.subheader("📊 Analysis Results")
                        st.write(analysis)
                    else:
                        st.error("❌ Failed to retrieve analysis.")

    # --------------------------- #
    #    Tab 2: Defect Information#
    # --------------------------- #
    with tab2:
        st.header("🔍 Defect Information")
        
        # Defect Comment Input
        defect_text = st.text_area(
            "📝 Enter Defect Comment/Narrative", 
            "Provide the defect description or inspection narrative here...",
            height=150
        )
        
        if len(defect_text) > 1000:
            st.warning("⚠️ Defect comment should be under 1000 characters.")
            defect_text = defect_text[:1000]
        
        # Analyze Button
        if st.button("🔍 Analyze Defect", key="analyze_defect"):
            if not defect_text.strip():
                st.warning("⚠️ Please enter a defect description to analyze.")
            else:
                with st.spinner("⏳ Analyzing defect information..."):
                    detailed_analysis = analyze_defect(defect_text, DEFECT_ANALYSIS_PROMPT, client)
                    if detailed_analysis:
                        st.subheader("📊 Detailed Breakdown")
                        st.write(detailed_analysis)
                    else:
                        st.error("❌ Failed to retrieve detailed analysis.")

# --------------------------- #
#        Execute App          #
# --------------------------- #

if __name__ == "__main__":
    main()
