import os
from io import BytesIO
import torch
import streamlit as st
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# -----------------------
# Page Config
# -----------------------
st.set_page_config(page_title="AI Image Generator", layout="centered")
st.title("🎨 AI Image Generation (MVP)")
st.write("Generate high-quality images from text prompts using Stable Diffusion.")

# -----------------------
# Hugging Face Token (from env variable or secret)
# -----------------------
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    st.error("Hugging Face token not found. Please set HUGGINGFACE_TOKEN environment variable or secret.")
    st.stop()

# -----------------------
# Load Stable Diffusion Pipeline
# -----------------------
@st.cache_resource(show_spinner=True)
def load_pipeline(hf_token):
    model_id = "runwayml/stable-diffusion-v1-5"
    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_auth_token=hf_token,
    )
    pipe.scheduler = scheduler

    # Move to GPU if available, else CPU
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")
    return pipe

pipe = load_pipeline(HF_TOKEN)

# -----------------------
# UI: Prompt and Options
# -----------------------
prompt = st.text_area("📝 Enter your prompt:", "A futuristic city at sunset, cyberpunk style, ultra-detailed")
steps = st.slider("Number of inference steps", 20, 50, 30)
guidance = st.slider("Guidance scale", 5.0, 12.0, 7.5)
size = st.radio("Image size", ["512x512", "768x768"])
width, height = map(int, size.split("x"))

# -----------------------
# Generate Button
# -----------------------
if st.button("🚀 Generate Image"):
    with st.spinner("Generating..."):
        try:
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
        except Exception as e:
            st.error(f"Error generating image: {e}")

# -----------------------
# Show Example Images (optional)
# -----------------------
st.markdown("---")
st.subheader("📸 Example Images")
example_dir = "examples"
if os.path.exists(example_dir):
    example_files = os.listdir(example_dir)
    for f in example_files:
        st.image(os.path.join(example_dir, f), use_column_width=True)
else:
    st.info("Add some example images in the 'examples' folder for the interview demo.")
