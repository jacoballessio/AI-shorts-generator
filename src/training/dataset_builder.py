"""
Dataset Builder - combines video labels and sources into training examples.
"""
import json
import re
from pathlib import Path
from typing import Dict, List


# Speech-only audio tags that are redundant once we have a real transcript
_SPEECH_TAGS = frozenset([
    'speech', 'voice', 'human voice', 'talking', 'speaking',
    'shout', 'yell', 'scream', 'whisper', 'singing',
])

# Minimum ratio of unique chars to total chars; below this = repeated-syllable garbage
_MIN_UNIQUE_RATIO = 0.15

# Garbage prefix tokens the audio captioner sometimes prepends to otherwise valid output
_GARBAGE_PREFIX_RE = re.compile(r'^(?:audles?|aud\s)\s*', re.IGNORECASE)


def _clean_audio_description(text: str) -> str | None:
    """
    Strip known garbage prefix tokens emitted by the audio captioner, then validate
    the remaining content. Returns the cleaned string, or None if entirely garbage.
    """
    if not text:
        return None
    t = text.strip()
    if len(t) < 3:
        return None
    # Strip known garbage prefix (e.g. "audles" prepended to valid content)
    stripped = _GARBAGE_PREFIX_RE.sub('', t).strip()
    was_prefixed = stripped != t
    if not stripped or len(stripped) < 3:
        return None  # Was only a garbage prefix with nothing meaningful after it
    # Capitalise first letter if a prefix was removed
    if was_prefixed:
        stripped = stripped[0].upper() + stripped[1:]
    s = stripped.lower()
    # Repetitive single-char content (e.g. "lelelele")
    if len(s) > 10 and len(set(s)) / len(s) < _MIN_UNIQUE_RATIO:
        return None
    # Repeating bigram/trigram run (e.g. "le" ×5)
    if re.search(r'(.{2,4})\1{4,}', s):
        return None
    return stripped


def _is_garbage_audio(text: str) -> bool:
    """Return True if the string cannot be salvaged as a valid audio description."""
    return _clean_audio_description(text) is None


def _extract_nonspeech_audio(desc: str) -> str:
    """
    Given a comma-separated audio tag string (e.g. "ic thunder, sidetone, human voice, speech"),
    return only the non-speech tags joined back as a string.
    Returns empty string if all tags are speech-related or if the description is
    primarily a sentence about speech/voice (e.g. "a man speaks over a radio").
    """
    # If the whole description is a sentence about speech, drop it entirely
    speech_sentence_phrases = [
        'speaks', 'speaking', 'talks', 'talking', 'says', 'saying',
        'a man speak', 'a woman speak', 'person speak', 'voice over',
    ]
    desc_lower = desc.lower()
    if any(p in desc_lower for p in speech_sentence_phrases):
        # Strip the speech part: try comma-split first, keep only non-speech tags
        tags = [t.strip() for t in desc.split(',') if t.strip()]
        nonspeech = [t for t in tags if not any(s in t.lower() for s in _SPEECH_TAGS)
                     and not any(p in t.lower() for p in speech_sentence_phrases)]
        return ', '.join(nonspeech)
    tags = [t.strip() for t in desc.split(',') if t.strip()]
    nonspeech = [t for t in tags if not any(s in t.lower() for s in _SPEECH_TAGS)]
    return ', '.join(nonspeech)


class DatasetBuilder:
    """Build training dataset from processed videos."""
    
    def __init__(self, output_dir: str = "data/training_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def build_training_example(
        self,
        video_metadata: Dict,
        labels: List[Dict],
        sources: List[Dict]
    ) -> Dict:
        """
        Create a single training example.
        
        Format:
        {
            "input": {
                "sources": [...],
                "duration": 514,
                "creator": "Veritasium",
                "topic": "Physics"
            },
            "output": {
                "segments": [...]
            }
        }
        """
        
        # Format sources for input.
        # Accept both scraped sources (have 'success' key) and raw detective sources
        # (NewsAPI results etc. that have a url + title but no 'success' key).
        formatted_sources = []
        for source in sources:
            # Skip if explicitly marked as failed
            if 'success' in source and not source['success']:
                continue
            url = source.get('url', '') or source.get('content', '')
            if not url.startswith('http'):
                continue
            formatted_sources.append({
                'title': source.get('title', '') or source.get('search_result_title', '') or '',
                'description': source.get('description', '') or source.get('summary', '') or '',
                'text': (source.get('text', '') or '')[:2000],
                'url': url,
                'type': source.get('source_type') or source.get('type', 'unknown'),
                'source_name': source.get('source_name', ''),
                'published_at': source.get('published_at', '') or source.get('publish_date', '') or '',
            })
        
        # Format output segments from transcript and labels
        # Get transcript segments with timestamps if available
        transcript_segments = video_metadata.get('transcript_segments', [])
        
        segments = self._create_segments(
            transcript_segments,
            labels,
            sources
        )
        # If ASID provided an overall audiovisual description, use it for audio_description only
        asid_desc = video_metadata.get('asid_description', '')
        if asid_desc and segments:
            nonspeech = _extract_nonspeech_audio(asid_desc)
            cleaned = _clean_audio_description(nonspeech)
            if cleaned:
                segments[0]['audio_description'] = cleaned
            # Speech-only tags are dropped — real transcript already captures what was said

        # If audio captioner provided per-segment descriptions, apply them
        seg_audio_descs = video_metadata.get('segment_audio_descriptions', [])
        if seg_audio_descs and segments:
            # zip and assign, don't overwrite existing values
            for segment, desc in zip(segments, seg_audio_descs):
                if desc and not segment.get('audio_description'):
                    nonspeech = _extract_nonspeech_audio(desc)
                    cleaned = _clean_audio_description(nonspeech)
                    if cleaned:
                        segment['audio_description'] = cleaned
        else:
            # Fallback: video-wide audio description (older behaviour)
            audio_desc = video_metadata.get('audio_description', '')
            if audio_desc and segments:
                nonspeech = _extract_nonspeech_audio(audio_desc)
                cleaned = _clean_audio_description(nonspeech)
                if cleaned:
                    for segment in segments:
                        if not segment.get('audio_description'):
                            segment['audio_description'] = cleaned

        
        training_example = {
            'input': {
                'sources': formatted_sources,
                'duration': video_metadata.get('duration', 0),
                'creator': video_metadata.get('channel', 'Unknown'),
                'topic': self._extract_topic(video_metadata),
                'title': video_metadata.get('title', ''),
            },
            'output': {
                'segments': segments,
            },
            'metadata': {
                'video_id': video_metadata.get('video_id'),
                'upload_date': video_metadata.get('upload_date'),
                'view_count': video_metadata.get('view_count'),
            }
        }
        
        return training_example
    
    def _create_segments(
        self,
        transcript_segments: List,
        labels: List[Dict],
        sources: List[Dict]
    ) -> List[Dict]:
        """Create time-aligned segments from transcript and visual labels."""
        
        segments = []
        
        if not labels:
            return segments
        
        # Group consecutive frames into clips based on visual similarity.
        # If the label carries explicit 'start_time'/'end_time' (written by
        # segment_based_analysis), each label is already a pre-decided window —
        # skip continuity merging entirely and convert 1:1 to clips.
        def _clip_start(lbl): return lbl.get('start_time', lbl['timestamp'])
        def _clip_end(lbl):   return lbl.get('end_time',   lbl['timestamp'])

        has_explicit_bounds = 'start_time' in labels[0] and labels[0]['start_time'] != labels[0].get('timestamp')

        if has_explicit_bounds:
            # Fast path: each label is its own clip (segment_based_analysis output)
            for label in labels:
                segments.append({
                    'start_time':  _clip_start(label),
                    'end_time':    _clip_end(label),
                    'visual_desc': label['description'],
                    'ui_elements': label.get('ui_elements', []),
                    'ui_html':     label.get('ui_html', ''),
                    'frame_count': 1,
                })
        else:
            # Legacy path: group per-frame labels by visual similarity
            current_clip = {
                'start_time': _clip_start(labels[0]),
                'end_time':   _clip_end(labels[0]),
                'visual_desc': labels[0]['description'],
                'ui_elements': labels[0].get('ui_elements', []),
                'ui_html': labels[0].get('ui_html', ''),
                'frame_count': 1,
            }
            
            for i, label in enumerate(labels[1:], 1):
                desc = label['description']
                
                # Check if this frame continues the same shot
                is_continuation = (
                    desc.lower().startswith('[continued:') or
                    desc.lower().startswith('same shot') or
                    desc == current_clip['visual_desc']
                )
                
                # Check for scene change marker
                has_scene_change = label.get('scene_change', False)
                
                if is_continuation and not has_scene_change:
                    # Extend current clip to the end of this label's window
                    current_clip['end_time'] = _clip_end(label)
                    current_clip['frame_count'] += 1
                    # Merge UI elements from this frame (dedupe later)
                    frame_ui = label.get('ui_elements', [])
                    if frame_ui and current_clip.get('ui_elements'):
                        current_clip['ui_elements'].extend(frame_ui)
                    elif frame_ui:
                        current_clip['ui_elements'] = frame_ui
                    
                    # Merge UI HTML from this frame
                    frame_ui_html = label.get('ui_html', '')
                    if frame_ui_html and current_clip.get('ui_html'):
                        current_clip['ui_html'] += '\n' + frame_ui_html
                    elif frame_ui_html:
                        current_clip['ui_html'] = frame_ui_html
                else:
                    # Save current clip and start new one
                    segments.append(current_clip)
                    current_clip = {
                        'start_time': _clip_start(label),
                        'end_time':   _clip_end(label),
                        'visual_desc': desc,
                        'ui_elements': label.get('ui_elements', []),
                        'ui_html': label.get('ui_html', ''),
                        'frame_count': 1,
                    }
            
            # Add final clip
            segments.append(current_clip)
        
        # Now create final segment format with combined descriptions
        final_segments = []
        for clip in segments:
            # Skip zero-duration clips (single frames)
            if clip['start_time'] == clip['end_time']:
                continue
            
            # Find transcript text for this time range
            clip_text = self._get_transcript_for_timerange(
                transcript_segments,
                clip['start_time'],
                clip['end_time']
            )
            
            # Extract UI elements from the clip if available
            ui_html = self._extract_ui_elements_from_clip(clip)
            
            # Create segment with visual description and transcript
            seg_dict = {
                'start_time': clip['start_time'],
                'end_time': clip['end_time'],
                'scene_description': clip['visual_desc'],
                'ui_elements_html': ui_html,
                'transcript': clip_text,
                'duration': clip['end_time'] - clip['start_time'],
            }
            # allow audio description placeholder if later added by build_training_example
            final_segments.append(seg_dict)
        
        # Post-process segments to close any gaps between them. If there is a gap
        # between the end of one segment and the start of the next, we adjust both
        # by half the gap duration so they butt up against each other. This helps
        # with downstream audio captioning and video generation where silent gaps
        # would otherwise be present.
        for i in range(len(final_segments) - 1):
            curr = final_segments[i]
            nxt = final_segments[i + 1]
            gap = nxt['start_time'] - curr['end_time']
            if gap > 0:
                adjust = gap / 2.0
                curr['end_time'] += adjust
                nxt['start_time'] -= adjust
                # recalc durations
                curr['duration'] = curr['end_time'] - curr['start_time']
                nxt['duration'] = nxt['end_time'] - nxt['start_time']
        return final_segments
    
    def _get_transcript_for_timerange(self, transcript_segments: List, start: float, end: float) -> str:
        """Extract transcript text for a time range using timestamp-based filtering.

        Each Whisper segment is attributed to the video segment whose time range
        contains the Whisper segment's midpoint.  This ensures no transcript text
        is shared across adjacent segments (which would cause duplication when a
        Whisper segment straddles a segment boundary).
        """
        if not transcript_segments:
            return ""
        
        relevant_text = []
        for seg in transcript_segments:
            # Handle both dict and object formats
            seg_start = seg.start_time if hasattr(seg, 'start_time') else seg.get('start_time', 0)
            seg_end = seg.end_time if hasattr(seg, 'end_time') else seg.get('end_time', 0)
            seg_text = seg.text if hasattr(seg, 'text') else seg.get('text', '')
            seg_speaker = seg.speaker if hasattr(seg, 'speaker') else seg.get('speaker')

            if seg_speaker and seg_text and not seg_text.lstrip().startswith('[SPEAKER_'):
                seg_text = f"[{seg_speaker}] {seg_text}"
            
            # Use the midpoint so each Whisper segment belongs to exactly one
            # video segment, even when it straddles a segment boundary.
            seg_mid = (seg_start + seg_end) / 2.0
            if start <= seg_mid < end:
                relevant_text.append(seg_text)
        
        return ' '.join(relevant_text).strip()
    
    def _extract_ui_elements_from_clip(self, clip: Dict) -> str:
        """Extract UI HTML from clip data."""
        # If the clip already has ui_html from VLM analysis, use it
        if 'ui_html' in clip and clip['ui_html']:
            return clip['ui_html']
        
        # Fallback: Convert UI elements to HTML representation
        ui_elements = clip.get('ui_elements', [])
        
        if not ui_elements:
            return ''
        
        # Deduplicate UI elements by (type, content, position)
        seen = set()
        unique_elements = []
        for elem in ui_elements:
            if not isinstance(elem, dict):
                continue
            key = (elem.get('type', ''), elem.get('content', ''), elem.get('position', ''))
            if key not in seen:
                seen.add(key)
                unique_elements.append(elem)
        
        html_parts = []
        for elem in unique_elements:
            elem_type = elem.get('type', 'text')
            content = elem.get('content', '')
            position = elem.get('position', '')
            style = elem.get('style', '')
            
            # Generate HTML based on element type
            if elem_type == 'text':
                classes = []
                if position:
                    classes.append(f'pos-{position.lower().replace(" ", "-")}')
                if style:
                    classes.append(f'style-{style.lower().replace(" ", "-")}')
                
                class_str = f' class="{" ".join(classes)}"' if classes else ''
                html_parts.append(f'<div{class_str}>{content}</div>')
            
            elif elem_type == 'icon':
                classes = ['icon']
                if position:
                    classes.append(f'pos-{position.lower().replace(" ", "-")}')
                class_str = f' class="{" ".join(classes)}"'
                html_parts.append(f'<span{class_str} aria-label="{content}"></span>')
            
            elif elem_type == 'graphic':
                classes = ['graphic']
                if position:
                    classes.append(f'pos-{position.lower().replace(" ", "-")}')
                class_str = f' class="{" ".join(classes)}"'
                html_parts.append(f'<div{class_str}>{content}</div>')
        
        return '\n'.join(html_parts)
    
    def _extract_topic(self, metadata: Dict) -> str:
        """Extract topic from video metadata."""
        
        categories = metadata.get('categories', [])
        tags = metadata.get('tags', [])
        
        if categories:
            return categories[0]
        elif tags:
            return tags[0]
        else:
            return 'General'

    def _load_labels_for_video(self, video_id: str, labels_dir: Path, analysis_dir: Path) -> List[Dict]:
        """Load labels for a video from the canonical labels dir or analysis cache."""
        labels_file = labels_dir / f"{video_id}.json"
        if labels_file.exists():
            with open(labels_file, 'r', encoding='utf-8') as f:
                labels = json.load(f)
            if isinstance(labels, list):
                return labels
            return []

        frames_path = analysis_dir / video_id / "frames_analysis.json"
        if not frames_path.exists():
            return []

        with open(frames_path, 'r', encoding='utf-8') as f:
            frames_data = json.load(f)

        if not isinstance(frames_data, list):
            return []

        labels: List[Dict] = []
        for frame in frames_data:
            if not isinstance(frame, dict):
                continue
            timestamp = frame.get('timestamp', 0)
            labels.append({
                'timestamp': timestamp,
                'start_time': frame.get('segment_start', timestamp),
                'end_time': frame.get('segment_end', timestamp),
                'description': frame.get('description', ''),
                'frame_number': frame.get('frame_number', 0),
                'text_visible': frame.get('text_visible', ''),
                'ui_elements': frame.get('ui_elements', []),
                'ui_html': frame.get('ui_html', ''),
            })
        return labels

    def load_saved_examples(self) -> List[Dict]:
        """Load already-written per-video examples from the training dataset directory."""
        dataset: List[Dict] = []
        for example_file in sorted(self.output_dir.glob("*_example.json")):
            try:
                with open(example_file, 'r', encoding='utf-8') as f:
                    example = json.load(f)
                if isinstance(example, dict) and 'input' in example and 'output' in example:
                    dataset.append(example)
            except Exception:
                continue
        return dataset
    
    def build_dataset(
        self,
        videos_dir: str = "data/raw_videos",
        labels_dir: str = "data/labels",
        sources_dir: str = "data/sources"
    ) -> List[Dict]:
        """Build complete dataset from all processed videos."""

        saved_examples = self.load_saved_examples()
        if saved_examples:
            return saved_examples
        
        videos_dir = Path(videos_dir)
        labels_dir = Path(labels_dir)
        sources_dir = Path(sources_dir)
        analysis_dir = videos_dir.parent / "analysis"
        
        dataset = []
        
        # Find all video metadata files
        metadata_files = list(videos_dir.glob("*.json"))
        
        for metadata_file in metadata_files:
            video_id = metadata_file.stem
            
            print(f"Processing {video_id}...")
            
            # Load metadata
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Load labels from canonical labels dir, or fall back to cached analysis.
            labels = self._load_labels_for_video(video_id, labels_dir, analysis_dir)
            if not labels:
                print(f"  No labels found, skipping")
                continue
            
            # Load sources
            sources_file = sources_dir / f"{video_id}_sources.json"
            if sources_file.exists():
                with open(sources_file, 'r', encoding='utf-8') as f:
                    sources = json.load(f)
            else:
                sources = []
            
            # Build training example
            example = self.build_training_example(metadata, labels, sources)
            dataset.append(example)
        
        return dataset
    
    def save_dataset(self, dataset: List[Dict], filename: str = "dataset.json"):
        """Save dataset to JSON file."""
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"\nDataset saved: {output_path}")
        print(f"Total examples: {len(dataset)}")
        
        # Print statistics
        total_sources = sum(len(ex['input']['sources']) for ex in dataset)
        total_segments = sum(len(ex['output']['segments']) for ex in dataset)
        
        print(f"Total sources: {total_sources}")
        print(f"Total segments: {total_segments}")
        print(f"Average sources per video: {total_sources/len(dataset):.1f}")
        print(f"Average segments per video: {total_segments/len(dataset):.1f}")


if __name__ == "__main__":
    # Example usage
    builder = DatasetBuilder(output_dir="data/training_dataset")
    
    # Build dataset from all processed videos
    # dataset = builder.build_dataset(
    #     videos_dir="data/raw_videos",
    #     labels_dir="data/labels",
    #     sources_dir="data/sources"
    # )
    
    # builder.save_dataset(dataset, "youtuber_dataset.json")
    
    print("Dataset Builder ready. Uncomment examples to test.")
