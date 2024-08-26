import requests
from PIL import Image
from io import BytesIO
import base64
import os
import moviepy.editor as mpy

# Define the API endpoints
txt2img_url = "http://127.0.0.1:7860/sdapi/v1/txt2img"
img2img_url = "http://127.0.0.1:7860/sdapi/v1/img2img"

# Define the folder to save images
output_folder = "nuclear_explosion_frames"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

def generate_initial_image():
    payload = {
        "prompt": "nuclear explosion, mushroom cloud, hyper realistic, cinematic lighting, detailed, immense energy release",
        "negative_prompt": "cartoon, illustration, painting, drawing, CGI",
        "steps": 35,
        "cfg_scale": 7.5,
        "width": 512,
        "height": 512,
        "sampler_index": "Euler a",
        "model": "realisticVisionV60B1_v51HyperVAE.safetensors"
    }

    response = requests.post(txt2img_url, json=payload)

    if response.status_code == 200:
        image_data = response.json()['images'][0]
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        image_path = os.path.join(output_folder, "initial_image.png")
        image.save(image_path)
        return image
    else:
        print(f"Failed to generate initial image: {response.status_code}")
        print(response.text)
        return None

def evolve_image(image, step):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    prompt = f"nuclear explosion, mushroom cloud expanding, step {step}, hyper realistic, immense energy release, cinematic lighting"

    payload = {
        "init_images": [img_base64],
        "prompt": prompt,
        "negative_prompt": "cartoon, illustration, painting, drawing, CGI",
        "steps": 35,
        "cfg_scale": 7.5,
        "width": 512,
        "height": 512,
        "denoising_strength": 0.5,  # Lowering this to keep more of the original image
        "model": "realisticVisionV60B1_v51HyperVAE.safetensors"
    }

    response = requests.post(img2img_url, json=payload)

    if response.status_code == 200:
        image_data = response.json()['images'][0]
        evolved_image = Image.open(BytesIO(base64.b64decode(image_data)))
        image_path = os.path.join(output_folder, f"evolved_image_{step}.png")
        evolved_image.save(image_path)
        return evolved_image
    else:
        print(f"Failed to evolve image: {response.status_code}")
        print(response.text)
        return None

def create_animation(frames, output_file, fps=12):
    clips = [mpy.ImageClip(frame).set_duration(1 / fps) for frame in frames]
    video = mpy.concatenate_videoclips(clips, method="compose")
    video.write_videofile(output_file, fps=fps)

def main():
    # Ask the user if they want to generate a new initial image or use the existing one
    use_existing_image = input("Do you want to use the existing initial image? (yes/no): ").strip().lower() == 'yes'
    
    # Ask the user for the video length in seconds
    video_length = int(input("Enter the length of the video in seconds: ").strip())
    fps = 12  # Frames per second
    total_frames = video_length * fps

    if use_existing_image and os.path.exists(os.path.join(output_folder, "initial_image.png")):
        initial_image = Image.open(os.path.join(output_folder, "initial_image.png"))
    else:
        initial_image = generate_initial_image()

    frames = []

    if initial_image:
        frames.append(os.path.join(output_folder, "initial_image.png"))

        for i in range(1, total_frames + 1):  # Generate frames based on the video length
            evolved_image = evolve_image(initial_image, i)
            if evolved_image:
                evolved_image_path = os.path.join(output_folder, f"evolved_image_{i}.png")
                frames.append(evolved_image_path)
                initial_image = evolved_image

        create_animation(frames, os.path.join(output_folder, "nuclear_explosion_animation.mp4"), fps=fps)

if __name__ == "__main__":
    main()
