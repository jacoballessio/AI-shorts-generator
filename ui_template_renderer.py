"""
UI Template Renderer
Generates realistic phone UI mockups from extracted UI elements.
Supports messaging apps (iMessage, WhatsApp), social media, and custom layouts.
"""

from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Optional, Tuple
import re
from pathlib import Path


class UITemplate:
    """Base class for UI templates."""
    
    def __init__(self, width: int = 1080, height: int = 1920):
        self.width = width
        self.height = height
        self.phones_aspect = width / height  # 0.5625 for 1080x1920 (portrait)
    
    def render(self, ui_elements: List[Dict]) -> Image.Image:
        """Render UI elements to image."""
        raise NotImplementedError


class MessagingUITemplate(UITemplate):
    """Realistic messaging app UI (iMessage, WhatsApp, etc.)."""
    
    COLORS = {
        'imessage': {
            'bg': (255, 255, 255),
            'sent': (0, 122, 255),      # Blue for sent messages
            'received': (229, 229, 234),  # Light gray for received
            'status_bar': (255, 255, 255),
            'header': (247, 247, 247),
            'text_sent': (255, 255, 255),
            'text_received': (0, 0, 0),
        },
        'whatsapp': {
            'bg': (230, 225, 215),
            'sent': (134, 199, 172),
            'received': (225, 245, 254),
            'status_bar': (25, 25, 25),
            'header': (77, 194, 98),
            'text_sent': (255, 255, 255),
            'text_received': (0, 0, 0),
        }
    }
    
    def __init__(self, width: int = 1080, height: int = 1920, app_type: str = 'imessage'):
        super().__init__(width, height)
        self.app_type = app_type.lower()
        if self.app_type not in self.COLORS:
            self.app_type = 'imessage'
        self.colors = self.COLORS[self.app_type]
    
    def _draw_status_bar(self, draw: ImageDraw.ImageDraw):
        """Draw iOS-style status bar."""
        y = 0
        height = 47
        
        # Background
        draw.rectangle([(0, y), (self.width, y + height)], 
                      fill=self.colors['status_bar'])
        
        # Draw status text (simplified)
        font_size = 28
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Time on left
        draw.text((40, y + 10), "9:41", fill=(0, 0, 0), font=font)
        
        # Battery/signal indicators on right (simplified)
        draw.text((self.width - 80, y + 10), "📶 🔋", fill=(0, 0, 0), font=font)
        
        return height
    
    def _draw_header(self, draw: ImageDraw.ImageDraw, app_name: str, y_offset: int):
        """Draw message header with app info."""
        header_height = 56
        
        # Header background
        draw.rectangle([(0, y_offset), (self.width, y_offset + header_height)],
                      fill=self.colors['header'])
        
        # App name
        font_size = 34
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        draw.text((20, y_offset + 12), app_name, fill=(0, 0, 0), font=font)
        
        return header_height
    
    def _extract_messages(self, ui_elements: List[Dict]) -> List[Tuple[str, str]]:
        """
        Extract messages from UI elements.
        Returns list of (message_text, sender_type) tuples.
        sender_type: 'sent', 'received', 'unknown'
        """
        messages = []
        
        for elem in ui_elements:
            content = elem.get('content', '').strip()
            if not content:
                continue
            
            # Detect sender based on position or styling
            position = elem.get('position', 'center')
            sender = 'received'  # Default
            
            if 'bottom' in position or 'right' in position:
                sender = 'sent'
            elif 'top' in position or 'left' in position:
                sender = 'received'
            
            # Filter out non-message content
            if not any(skip in content.lower() for skip in 
                      ['time', 'battery', 'signal', ':', '@', 'message']):
                if len(content) > 1:  # Skip single characters
                    messages.append((content, sender))
        
        return messages
    
    def _wrap_text(self, text: str, font, max_width: int) -> List[str]:
        """Wrap text to fit width."""
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = font.getbbox(test_line)
            line_width = bbox[2] - bbox[0]
            
            if line_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    def _draw_message_bubble(self, draw: ImageDraw.ImageDraw, text: str, 
                            x: int, y: int, sender: str, font) -> int:
        """
        Draw a single message bubble.
        Returns the height used.
        """
        # Bubble styling
        padding = 12
        margin = 20
        max_width = self.width - 2 * margin - 40
        
        # Wrap text
        lines = self._wrap_text(text, font, max_width)
        
        # Calculate bubble size
        line_height = 24
        bubble_height = len(lines) * line_height + 2 * padding
        
        # Bubble position
        if sender == 'sent':
            bubble_x = self.width - max_width - margin - padding
            bubble_color = self.colors['sent']
            text_color = self.colors['text_sent']
            text_align = 'right'
        else:
            bubble_x = margin
            bubble_color = self.colors['received']
            text_color = self.colors['text_received']
            text_align = 'left'
        
        # Draw rounded rectangle (simplified with rectangle for now)
        bubble_y = y
        bubble_width = max_width
        
        # Draw bubble
        draw.rounded_rectangle(
            [(bubble_x, bubble_y), (bubble_x + bubble_width, bubble_y + bubble_height)],
            radius=18,
            fill=bubble_color,
            outline=bubble_color
        )
        
        # Draw text
        text_y = bubble_y + padding
        for line in lines:
            if text_align == 'right':
                text_x = bubble_x + bubble_width - padding
                draw.text((text_x, text_y), line, fill=text_color, font=font, anchor='rt')
            else:
                text_x = bubble_x + padding
                draw.text((text_x, text_y), line, fill=text_color, font=font, anchor='lt')
            text_y += line_height
        
        return bubble_height + 12  # Add spacing between messages
    
    def render(self, ui_elements: List[Dict], app_name: str = 'iMessage') -> Image.Image:
        """Render a messaging UI mockup."""
        # Create image
        img = Image.new('RGB', (self.width, self.height), color=self.colors['bg'])
        draw = ImageDraw.Draw(img)
        
        # Load fonts
        try:
            header_font = ImageFont.truetype("arial.ttf", 34)
            message_font = ImageFont.truetype("arial.ttf", 32)
            time_font = ImageFont.truetype("arial.ttf", 24)
        except:
            header_font = ImageFont.load_default()
            message_font = ImageFont.load_default()
            time_font = ImageFont.load_default()
        
        # Draw status bar
        y = self._draw_status_bar(draw)
        
        # Draw header
        y += self._draw_header(draw, app_name, y)
        
        # Add padding
        y += 20
        
        # Extract and draw messages
        messages = self._extract_messages(ui_elements)
        
        if not messages:
            # Fallback: show raw elements
            for elem in ui_elements:
                content = elem.get('content', '').strip()
                if content and len(content) > 1:
                    bubble_height = self._draw_message_bubble(
                        draw, content, 0, y, 'received', message_font
                    )
                    y += bubble_height + 8
                    
                    if y > self.height - 100:
                        break
        else:
            # Draw extracted messages
            for message, sender in messages:
                if self.app_type == 'imessage':
                    # Alternate sender for iMessage
                    sender = 'sent' if messages.index((message, sender)) % 2 == 0 else 'received'
                
                bubble_height = self._draw_message_bubble(
                    draw, message, 0, y, sender, message_font
                )
                y += bubble_height + 8
                
                if y > self.height - 100:
                    break
        
        # Draw input area at bottom
        input_y = self.height - 80
        draw.rectangle([(0, input_y - 1), (self.width, input_y)],
                      fill=(200, 200, 200))  # Separator line
        
        # Input box
        input_height = 50
        draw.rounded_rectangle(
            [(20, input_y + 15), (self.width - 20, input_y + 15 + input_height)],
            radius=22,
            fill=(240, 240, 240),
            outline=(200, 200, 200)
        )
        
        # Placeholder text
        draw.text((40, input_y + 28), "Message...", 
                 fill=(150, 150, 150), font=message_font)
        
        return img


class GenericUITemplate(UITemplate):
    """Generic template for non-messaging UIs."""
    
    def render(self, ui_elements: List[Dict]) -> Image.Image:
        """Render generic UI with stacked elements."""
        img = Image.new('RGB', (self.width, self.height), color=(240, 240, 245))
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 40)
            font_small = ImageFont.truetype("arial.ttf", 30)
        except:
            font = ImageFont.load_default()
            font_small = font
        
        # Group by position
        from collections import defaultdict
        position_groups = defaultdict(list)
        
        for elem in ui_elements:
            content = elem.get('content', '').strip()
            if not content:
                continue
            
            position = elem.get('position', 'center')
            position_groups[position].append((content, elem.get('style', '')))
        
        # Draw each position group
        for position, elements in position_groups.items():
            if position == 'top':
                y = 100
                x = self.width // 2
            elif position == 'bottom':
                y = self.height - 300
                x = self.width // 2
            elif position == 'top-left':
                y = 80
                x = 80
            elif position == 'top-right':
                y = 80
                x = self.width - 80
            else:
                y = self.height // 2
                x = self.width // 2
            
            for content, style in elements:
                draw.text((x, y), content, fill=(50, 50, 50), font=font_small, anchor='mm')
                y += 50


def detect_app_type(ui_elements: List[Dict], scene_description: str) -> str:
    """Detect app type from UI elements and scene description."""
    # combined = ' '.join([
    #     scene_description,
    #     ' '.join([elem.get('content', '') for elem in ui_elements])
    # ]).lower()
    
    # if 'imessage' in combined or 'message' in combined and 'iphone' in combined:
    #     return 'imessage'
    # elif 'whatsapp' in combined:
    #     return 'whatsapp'
    # elif 'sms' in combined or 'text' in combined:
    #     return 'imessage'  # Close enough
    # Broken --- fix later
    
    return 'generic'


def render_ui_template(ui_elements: List[Dict], scene_description: str = '', 
                      width: int = 1080, height: int = 1920) -> Image.Image:
    """
    Render UI elements using appropriate template.
    
    Args:
        ui_elements: List of UI element dicts with 'content', 'position', 'style'
        scene_description: Scene context for app detection
        width: Output image width
        height: Output image height
    
    Returns:
        PIL Image with rendered UI
    """
    app_type = detect_app_type(ui_elements, scene_description)
    
    if app_type in ['imessage', 'whatsapp']:
        template = MessagingUITemplate(width, height, app_type)
        app_name = 'iMessage' if app_type == 'imessage' else 'WhatsApp'
        return template.render(ui_elements, app_name)
    else:
        template = GenericUITemplate(width, height)
        return template.render(ui_elements)
