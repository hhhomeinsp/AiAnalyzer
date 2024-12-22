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

# Get API key from environment or secrets
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except:
    # Fallback to environment variable for local development
    openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OpenAI API key is not configured")
    st.stop()

# Initialize OpenAI client
client = OpenAI(
    api_key=openai_api_key,
    base_url="https://api.openai.com/v1"
)

# Also set for older openai package usage
openai.api_key = openai_api_key

# Define AI Prompts as Constants
IMAGE_ANALYSIS_PROMPT = (
    "Please analyze the given image along with any provided text context (if any) and provide an analysis "
    "of any deficiencies or conditions, safety concerns, functionality issues, etc."
)

DEFECT_ANALYSIS_PROMPT = (
    "Please analyze the given deficiency comment and provide a more detailed breakdown of the comment "
    "to allow for better understanding."
)

# Configure logging
logging.basicConfig(level=logging.INFO)

# --------------------------- #
#        AI Functions         #
# --------------------------- #

def analyze_image(image_file, context, prompt):
    """
    Analyze image
    
    Parameters:
        image_file (UploadedFile): The image file uploaded by the user.
        context (str): Additional text context provided by the user.
        prompt (str): The AI prompt guiding the analysis.
        
    Returns:
        str: The AI-generated analysis or an error message.
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
        image.save(buffered, format="JPEG", quality=70)  # Adjust quality as needed
        
        # Get the byte data
        image_bytes = buffered.getvalue()
        
        # Encode to Base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return f"Error processing image: {str(e)}"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
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

def analyze_defect(defect_text, prompt):
    """
    Analyze defect comment
    
    Parameters:
        defect_text (str): The defect comment/narrative provided by the user.
        prompt (str): The AI prompt guiding the analysis.
        
    Returns:
        str: The AI-generated detailed breakdown or an error message.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
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
    # Set Streamlit page configuration
    st.set_page_config(page_title="Home Inspection AI Assistant", layout="wide")
    st.title("🏠 Home Inspection AI Assistant")

    # Provide informational message
    st.info("📌 Please ensure that your images are clear and within the 5MB size limit for optimal analysis.")

    # Create two tabs: Image Analysis and Defect Information
    tab1, tab2 = st.tabs(["🖼️ Image Analysis", "🔍 Defect Information"])

    # --------------------------- #
    #      Tab 1: Image Analysis   #
    # --------------------------- #
    with tab1:
        st.header("🖼️ Image Analysis")
        
        # File Uploader for Image with Size Limit (e.g., 5MB)
        uploaded_file = st.file_uploader("📤 Upload an image (Max 5MB)", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            # Check file size
            uploaded_file.seek(0, os.SEEK_END)
            file_size = uploaded_file.tell()
            uploaded_file.seek(0)  # Reset file pointer
            
            if file_size > 5 * 1024 * 1024:  # 5MB limit
                st.error("❌ The uploaded image exceeds the 5MB size limit. Please upload a smaller image.")
            else:
                try:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="📸 Uploaded Image", use_column_width=True)
                except Exception as e:
                    st.error(f"Error displaying image: {str(e)}")
        
        # Additional Context Input with Character Limit
        context = st.text_area(
            "📝 Additional Context (Optional)", 
            "Provide any relevant context about the image (e.g., location, age of home, specific concerns).",
            height=100
        )
        
        if len(context) > 500:
            st.warning("⚠️ Additional context should be under 500 characters.")
            context = context[:500]
        
        # Analyze Button
        if st.button("🔍 Analyze Image"):
            if uploaded_file is None:
                st.warning("⚠️ Please upload an image to analyze.")
            else:
                with st.spinner("⏳ Analyzing image..."):
                    analysis = analyze_image(uploaded_file, context, IMAGE_ANALYSIS_PROMPT)
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
        
        # Defect Comment Input with Character Limit
        defect_text = st.text_area(
            "📝 Enter Defect Comment/Narrative", 
            "Provide the defect description or inspection narrative here...",
            height=150
        )
        
        if len(defect_text) > 1000:
            st.warning("⚠️ Defect comment should be under 1000 characters.")
            defect_text = defect_text[:1000]
        
        # Analyze Button
        if st.button("🔍 Analyze Defect"):
            if not defect_text.strip():
                st.warning("⚠️ Please enter a defect description to analyze.")
            else:
                with st.spinner("⏳ Analyzing defect information..."):
                    detailed_analysis = analyze_defect(defect_text, DEFECT_ANALYSIS_PROMPT)
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
