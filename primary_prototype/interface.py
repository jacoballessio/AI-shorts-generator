import os
import gradio as gr
from main import main
from werkzeug.utils import secure_filename

def generate_video_shorts(video_file, youtube_url, shorts_length, visual_information_density, user_prompt, cache_dir, cache_shorts):
    # Define a variable to store progress or other messages
    progress_message = "Starting the video shorts generation...\n"
    total_cost = 0  # Placeholder for the total cost returned by main()

    # Get the original file name
    video_path = None
    if video_file is not None:
        if isinstance(video_file, str):
            # If video_file is a path (string), use it directly
            original_filename = os.path.basename(video_file)
            original_directory = os.path.dirname(video_file)
            new_filename = original_filename.replace(" ", "_")
            video_path = os.path.join(original_directory, new_filename)

            # Rename the file to the new path with no spaces
            if os.path.exists(video_path):
                progress_message += f"File {new_filename} already exists.\n"
            else:
                os.rename(video_file, video_path)
                progress_message += f"File renamed to {new_filename}\n"
        else:
            # Remove spaces from the original file name
            original_filename = video_file.name
            original_filename_no_spaces = original_filename.replace(" ", "_")

            # Generate a secure version of the file name
            secure_filename_no_ext = os.path.splitext(secure_filename(original_filename_no_spaces))[0]
            secure_filename_with_ext = secure_filename_no_ext + ".mp4"

            # Create a secure temporary directory
            temp_dir = os.path.join(cache_dir, "temp")
            os.makedirs(temp_dir, exist_ok=True)

            # Construct the file path using the secure file name
            video_path = os.path.join(temp_dir, secure_filename_with_ext)

            # Save the uploaded video file with the secure file name
            with open(video_path, "wb") as output_file:
                output_file.write(video_file)  # Use video_file directly

            progress_message += f"Video file saved as {secure_filename_with_ext}.\n"
    
    progress_message += "Extracting audio...\n"
    
    if video_path is not None:
        print(video_path)
    
    # Assuming the `main` function now returns a total_cost value
    total_cost = main(shorts_length, visual_information_density, user_prompt, cache_dir, cache_shorts, video_path=video_path, youtube_url=youtube_url)

    final_shorts_path = "final_video_with_audio.mp4"
    
    # Check if the final video file exists
    if os.path.exists(final_shorts_path):
        progress_message += "Video shorts generated successfully!\n"
        final_output_video = gr.Video(value=final_shorts_path, label="Generated Video Shorts")
    else:
        progress_message += "Failed to generate video shorts. Please check the console for error messages.\n"
        final_output_video = gr.Video(label="Generated Video Shorts")  # Create an empty Video component

    return progress_message, final_output_video, f"Total Cost: ${total_cost:.2f}"

# Create a Gradio interface
iface = gr.Interface(
    fn=generate_video_shorts,
    inputs=[
        gr.File(label="Input Video"),
        gr.Textbox(label="Or enter youtube URL"),
        gr.Number(value=15, label="Shorts Length (seconds)"),
        gr.Slider(value=15, minimum=0, maximum=100, label="Visual Information Density"),
        gr.Textbox(value="energetic style, warm color grading, spoken by pirate", label="Custom Instructions"),
        gr.Textbox(value="J:/temp", label="Cache Directory"),
        gr.Checkbox(value=True, label="Cache Shorts"),
    ],
    outputs=[
        gr.Textbox(label="Generation Progress"),
        gr.Video(visible=True),  # Placeholder output component
        gr.Textbox(label="Total Cost"),  # New output for displaying total cost
    ],
    title="AI Video Shorts Generator",
    description="Upload a video and generate engaging video shorts.",
    allow_flagging="never",
)

# Launch the Gradio interface with sharing enabled
iface.launch(share=True)
