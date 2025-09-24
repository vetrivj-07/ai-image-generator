import os
from io import BytesIO
import torch
import streamlit as st
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# Page config
st.set_page_config(page_title="AI Image Generator", layout="centered")
st.title("🎨 AI Image Generation (MVP)")
st.write("Generate high-quality images from text prompts using Stable Diffusion.")

# Hugging Face token
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    HF_TOKEN = st.text_input("Enter your Hugging Face token:", type="password")
    if not HF_TOKEN:
        st.stop()

# Load pipeline
@st.cache_resource(show_spinner=False)
def load_pipeline(hf_token):
    model_id = "runwayml/stable-diffusion-v1-5"
    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_auth_token=hf_token,
    )
    pipe.scheduler = scheduler
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    return pipe

pipe = load_pipeline(HF_TOKEN)

# Prompt input
prompt = st.text_area("📝 Enter your prompt:", "A futuristic city at sunset, cyberpunk style, ultra-detailed")
steps = st.slider("Number of inference steps", 20, 50, 30)
guidance = st.slider("Guidance scale", 5.0, 12.0, 7.5)
size = st.radio("Image size", ["512x512", "768x768"])
width, height = map(int, size.split("x"))

# Generate button
if st.button("🚀 Generate Image"):
    with st.spinner("Generating..."):
        result = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance,
        )
        image = result.images[0]

        # Show image
        st.image(image, caption="Generated Image", use_column_width=True)

        # Download option
        buf = BytesIO()
        image.save(buf, format="PNG")
        st.download_button(
            "Download Image",
            data=buf.getvalue(),
            file_name="generated.png",
            mime="image/png"
        )
