import os
import gradio as gr
from main import main
from werkzeug.utils import secure_filename


def generate_video_shorts(video_file, shorts_length, cache_dir, cache_shorts):
    # Get the original file name
    original_filename = video_file.name

    # Generate a secure version of the file name
    secure_filename_no_ext = os.path.splitext(secure_filename(original_filename))[0]
    secure_filename_with_ext = secure_filename_no_ext + ".mp4"

    # Create the "temp" directory if it doesn't exist
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)

    # Construct the file path using the secure file name
    video_path = os.path.join(temp_dir, secure_filename_with_ext)

    # Save the uploaded video file with the secure file name
    with open(video_file.name, "rb") as input_file:
        with open(video_path, "wb") as output_file:
            output_file.write(input_file.read())
    
    # Create a Gradio output component for displaying updates
    output_text = gr.Textbox(label="Generation Progress")

    # Create a temporary Gradio output component for displaying the final video
    output_video = gr.Video(label="Generated Video Shorts")

    output_text.value = "Extracting audio..."
    main(video_path, shorts_length, cache_dir, cache_shorts)

    final_shorts_path = "final_video_with_audio.mp4"
    
    # Check if the final video file exists
    if os.path.exists(final_shorts_path):
        # Create a new Gradio output component for displaying the final video
        final_output_video = gr.Video(value=final_shorts_path, label="Generated Video Shorts")
        output_text.value += "\n\nVideo shorts generated successfully!"
    else:
        final_output_video = gr.Video(label="Generated Video Shorts") # Create an empty Video component
        output_text.value += "\n\nFailed to generate video shorts. Please check the console for error messages."

    return [output_text, final_output_video]

# Create a Gradio interface
iface = gr.Interface(
    fn=generate_video_shorts,
    inputs=[
        gr.File(label="Input Video"),
        gr.Number(value=15, label="Shorts Length (seconds)"),
        gr.Textbox(value="J:/temp", label="Cache Directory"),
        gr.Checkbox(value=True, label="Cache Shorts"),
    ],
    outputs=[
        gr.Textbox(label="Generation Progress"),
        gr.Video(visible=True),  # Placeholder output component
    ],
    title="AI Video Shorts Generator",
    description="Upload a video and generate engaging video shorts.",
    allow_flagging="never",
)

# Launch the Gradio interface with sharing enabled
iface.launch(share=True)