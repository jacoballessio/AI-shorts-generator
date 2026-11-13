"""
QA Interface - manual review and correction of detected sources.
"""
import json
import gradio as gr
from pathlib import Path
from typing import Dict, List, Optional


class QAInterface:
    """Interactive interface for reviewing and correcting source detection."""
    
    def __init__(
        self,
        videos_dir: str = "data/raw_videos",
        labels_dir: str = "data/labels",
        sources_dir: str = "data/sources"
    ):
        self.videos_dir = Path(videos_dir)
        self.labels_dir = Path(labels_dir)
        self.sources_dir = Path(sources_dir)
        
        # Load all video IDs
        self.video_ids = [f.stem for f in self.videos_dir.glob("*.json")]
        self.current_index = 0
        
    def load_video_data(self, video_id: str) -> Dict:
        """Load all data for a video."""
        
        # Load metadata
        with open(self.videos_dir / f"{video_id}.json", 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Load labels
        labels_file = self.labels_dir / f"{video_id}.json"
        labels = []
        if labels_file.exists():
            with open(labels_file, 'r', encoding='utf-8') as f:
                labels = json.load(f)
        
        # Load sources
        sources_file = self.sources_dir / f"{video_id}_sources.json"
        sources = []
        if sources_file.exists():
            with open(sources_file, 'r', encoding='utf-8') as f:
                sources = json.load(f)
        
        return {
            'metadata': metadata,
            'labels': labels,
            'sources': sources,
        }
    
    def save_corrected_sources(self, video_id: str, sources: List[Dict]):
        """Save manually corrected sources."""
        
        output_file = self.sources_dir / f"{video_id}_sources_corrected.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sources, f, indent=2, ensure_ascii=False)
        
        return f"Saved to {output_file}"
    
    def create_interface(self):
        """Create Gradio interface for QA."""
        
        with gr.Blocks(title="Source Detective QA") as demo:
            gr.Markdown("# 🔍 Source Detective QA Interface")
            gr.Markdown("Review and correct detected sources for training data quality.")
            
            with gr.Row():
                with gr.Column(scale=1):
                    video_selector = gr.Dropdown(
                        choices=self.video_ids,
                        label="Select Video",
                        value=self.video_ids[0] if self.video_ids else None
                    )
                    load_btn = gr.Button("Load Video Data", variant="primary")
                
                with gr.Column(scale=2):
                    video_info = gr.Textbox(
                        label="Video Info",
                        lines=3,
                        interactive=False
                    )
            
            gr.Markdown("## Transcript & Description")
            
            with gr.Row():
                transcript_box = gr.Textbox(
                    label="Transcript",
                    lines=10,
                    interactive=False
                )
                description_box = gr.Textbox(
                    label="Description",
                    lines=10,
                    interactive=False
                )
            
            gr.Markdown("## Detected Sources")
            
            sources_editor = gr.JSON(
                label="Sources (editable)",
                value=[]
            )
            
            with gr.Row():
                verdict = gr.Radio(
                    choices=["✅ Approve All", "⚠️ Needs Review", "❌ Reject"],
                    label="Quality Verdict"
                )
                save_btn = gr.Button("Save Corrections", variant="primary")
            
            status = gr.Textbox(label="Status", interactive=False)
            
            # Event handlers
            def load_video(video_id):
                if not video_id:
                    return "", "", "", []
                
                data = self.load_video_data(video_id)
                metadata = data['metadata']
                
                info = f"""Title: {metadata.get('title', 'N/A')}
Channel: {metadata.get('channel', 'N/A')}
Duration: {metadata.get('duration', 0)}s"""
                
                transcript = metadata.get('transcript', 'No transcript available')
                description = metadata.get('description', 'No description')
                sources = data['sources']
                
                return info, transcript, description, sources
            
            def save_corrections(video_id, sources, verdict):
                if not video_id:
                    return "No video selected"
                
                # Add verdict to sources
                for source in sources:
                    source['qa_verdict'] = verdict
                
                return self.save_corrected_sources(video_id, sources)
            
            load_btn.click(
                fn=load_video,
                inputs=[video_selector],
                outputs=[video_info, transcript_box, description_box, sources_editor]
            )
            
            save_btn.click(
                fn=save_corrections,
                inputs=[video_selector, sources_editor, verdict],
                outputs=[status]
            )
        
        return demo
    
    def launch(self, share: bool = False):
        """Launch the QA interface."""
        
        demo = self.create_interface()
        demo.launch(share=share, server_name="0.0.0.0", server_port=7861)


if __name__ == "__main__":
    # Launch QA interface
    qa = QAInterface(
        videos_dir="data/raw_videos",
        labels_dir="data/labels",
        sources_dir="data/sources"
    )
    
    # qa.launch(share=False)
    
    print("QA Interface ready. Uncomment to launch.")
