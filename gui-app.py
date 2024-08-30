import os
import requests
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib
import matplotlib.pyplot as plt
import tkinter as tk
from threading import Thread, Lock, Event
from queue import Queue, Empty
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Ensure matplotlib uses the TkAgg backend
matplotlib.use("TkAgg")

# Stable Diffusion API URL
api_url = "http://127.0.0.1:7860/sdapi/v1/img2img"

# Create a directory for saving images if it doesn't exist
output_dir = "images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Global variables
previous_generated_image = None
main_prompt = "A portrait of an anime character"
morph_prompt = "A colorful abstract pattern"
negative_prompt = "distorted, ugly, blurry"
image_counter = 0
paused = False
lock = Lock()
task_queue = Queue()
stop_event = Event()  # Event to control pausing and stopping threads

# Function to create a random 64x64 image
def create_random_image(size=(512, 480)):
    array = np.random.randint(0, 256, size + (3,), dtype=np.uint8)
    return Image.fromarray(array)

# Function to send the image to Stable Diffusion API
def process_image_with_stable_diffusion(image, init_image=None):
    global image_counter
    try:
        # Convert the image to a format compatible with the API
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        init_img_str = None
        if init_image:
            buffered_init = BytesIO()
            init_image.save(buffered_init, format="PNG")
            init_img_str = base64.b64encode(buffered_init.getvalue()).decode()

        payload = {
            "init_images": [init_img_str] if init_img_str else [img_str],
            "prompt": f"{main_prompt}, {morph_prompt}",
            "negative_prompt": negative_prompt,
            "strength": 0.6 if init_img_str else 0.75,
            "steps": 20,
        }

        response = requests.post(api_url, json=payload)

        if response.status_code == 200:
            response_data = response.json()
            img_data = base64.b64decode(response_data['images'][0])
            generated_image = Image.open(BytesIO(img_data))
            
            image_counter += 1
            generated_image.save(os.path.join(output_dir, f"image_{image_counter:04d}.png"))
            
            return generated_image
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Exception occurred: {e}")
        return None

# Function to update the matplotlib figure
def update_frame(ax):
    global previous_generated_image, paused

    while not stop_event.is_set():  # Continue running unless the stop_event is set
        if paused or stop_event.is_set():
            continue

        try:
            random_image = create_random_image()
            task_queue.put(random_image)
            break  # Yield control after adding a task
        except Exception as e:
            print(f"Exception in update_frame: {e}")
            break

# Function to process tasks from the queue
def process_task_queue(ax):
    global previous_generated_image

    while not stop_event.is_set():  # Continue running unless the stop_event is set
        if stop_event.is_set():
            break  # Break out of the loop if we need to stop

        try:
            random_image = task_queue.get(timeout=0.1)  # Non-blocking get with timeout
            if stop_event.is_set():
                break  # Break out if stop is set after retrieving a task
            generated_image = process_image_with_stable_diffusion(random_image, init_image=previous_generated_image)

            if generated_image and not stop_event.is_set():
                previous_generated_image = generated_image

                ax.imshow(generated_image)
                plt.draw()
            task_queue.task_done()
        except Empty:
            pass  # No task to process, yield control

# Function to start the threads
def start_threads(ax):
    stop_event.clear()  # Ensure the stop event is cleared when starting threads
    Thread(target=process_task_queue, args=(ax,), daemon=True).start()

# Function to update the morph prompt from the textbox on key release
def update_morph_prompt(event=None):
    global morph_prompt
    morph_prompt = morph_entry.get()

# Function to update the main prompt from the textbox on key release
def update_main_prompt(event=None):
    global main_prompt
    main_prompt = main_entry.get()

# Function to update the negative prompt from the textbox on key release
def update_negative_prompt(event=None):
    global negative_prompt
    negative_prompt = negative_entry.get()

# Function to toggle pause/resume
def toggle_pause():
    global paused
    with lock:
        paused = not paused
        if paused:
            stop_event.set()  # Stop the threads
            with task_queue.mutex:
                task_queue.queue.clear()  # Clear any remaining tasks in the queue
        else:
            stop_event.clear()  # Clear the stop event
            start_threads(ax)  # Restart the threads
    pause_button.config(text="Resume" if paused else "Pause")

# Initialize tkinter for GUI
root = tk.Tk()
root.title("Stable Diffusion 64x64 Morphing")

# Create a textbox for the main prompt
main_prompt_label = tk.Label(root, text="Main Prompt:")
main_prompt_label.pack()

main_entry = tk.Entry(root, width=50)
main_entry.pack()
main_entry.insert(0, main_prompt)
main_entry.bind("<KeyRelease>", update_main_prompt)

# Create a textbox for the morph prompt
morph_prompt_label = tk.Label(root, text="Morph Prompt:")
morph_prompt_label.pack()

morph_entry = tk.Entry(root, width=50)
morph_entry.pack()
morph_entry.insert(0, morph_prompt)
morph_entry.bind("<KeyRelease>", update_morph_prompt)

# Create a textbox for the negative prompt
negative_prompt_label = tk.Label(root, text="Negative Prompt:")
negative_prompt_label.pack()

negative_entry = tk.Entry(root, width=50)
negative_entry.pack()
negative_entry.insert(0, negative_prompt)
negative_entry.bind("<KeyRelease>", update_negative_prompt)

# Create a pause button
pause_button = tk.Button(root, text="Pause", command=toggle_pause)
pause_button.pack()

# Create a matplotlib figure for display
fig, ax = plt.subplots()
ax.axis('off')  # Turn off the axis

# Start the threads
start_threads(ax)

# Embed the matplotlib plot in the tkinter window using FigureCanvasTkAgg
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Set up a loop to update frames periodically
def periodic_update():
    if not paused:
        update_frame(ax)
    root.after(1000, periodic_update)  # Update every 1 second

root.after(1000, periodic_update)  # Start the periodic update
root.mainloop()
