from typing import Dict, List, Tuple
import random
import logging

logger = logging.getLogger(__name__)

def get_writing_style(style_name: str) -> 'WritingStyles':
    """Get a writing style instance by name."""
    return WritingStyles().get_style(style_name)


class WritingStyles:
    """Enhanced writing styles for more engaging content."""
    
    def __init__(self):
        self.styles = {
            'storytelling': {
                'intro_templates': [
                    "Picture this scenario: {scenario}",
                    "Let me tell you a story about {topic}",
                    "It all started when {scenario}",
                    "Imagine a world where {topic} changes everything",
                    "Here's an interesting tale about {topic}"
                ],
                'transitions': [
                    "But that's not even the best part...",
                    "Here's where things get interesting...",
                    "Now, you might be wondering what happened next...",
                    "This is where the story takes an unexpected turn...",
                    "And just when you think that's all..."
                ],
                'conclusions': [
                    "And that's how {topic} changed everything",
                    "The moral of this story? {lesson}",
                    "Looking back, it's clear that {insight}",
                    "This story shows us that {lesson}",
                    "What we learned from this experience is {insight}"
                ]
            },
            'expert_analysis': {
                'intro_templates': [
                    "Based on recent developments in {topic}...",
                    "After analyzing the latest trends in {topic}...",
                    "From a technical perspective, {topic} presents...",
                    "Let's dive deep into the world of {topic}...",
                    "A comprehensive analysis of {topic} reveals..."
                ],
                'evidence_phrases': [
                    "Research shows that...",
                    "According to industry experts...",
                    "Recent studies indicate...",
                    "Data analysis reveals...",
                    "Expert consensus suggests..."
                ],
                'analysis_frames': [
                    "Let's break this down into key components...",
                    "There are several critical factors to consider...",
                    "From a strategic perspective...",
                    "When we examine the data closely...",
                    "The analysis can be approached from multiple angles..."
                ]
            },
            'tutorial': {
                'intro_templates': [
                    "Ready to master {topic}? Let's dive in!",
                    "In this step-by-step guide to {topic}...",
                    "Everything you need to know about {topic}",
                    "Master {topic} with this comprehensive guide",
                    "Learn {topic} the right way with these steps"
                ],
                'step_formats': [
                    "Step {num}: {step}",
                    "Part {num}: {step}",
                    "Phase {num}: {step}",
                    "Stage {num}: {step}",
                    "#{num}: {step}"
                ],
                'tips_formats': [
                    "Pro Tip: {tip}",
                    "Quick Tip: {tip}",
                    "Expert Advice: {tip}",
                    "Important Note: {tip}",
                    "Remember: {tip}"
                ]
            },
            'debate': {
                'intro_templates': [
                    "The controversy surrounding {topic} raises important questions...",
                    "There are two sides to every story, and {topic} is no exception...",
                    "Let's explore the ongoing debate about {topic}",
                    "The {topic} debate continues to divide opinions...",
                    "When it comes to {topic}, opinions are sharply divided..."
                ],
                'perspective_transitions': [
                    "On the other hand...",
                    "Proponents argue that...",
                    "Critics point out that...",
                    "Supporters maintain that...",
                    "Skeptics counter with..."
                ],
                'balanced_conclusions': [
                    "While both sides make valid points...",
                    "The truth likely lies somewhere in between...",
                    "Moving forward, it's important to consider...",
                    "A balanced approach suggests...",
                    "The evidence points to a middle ground..."
                ]
            },
            'trend_analysis': {
                'intro_templates': [
                    "The latest trends in {topic} reveal...",
                    "How is {topic} evolving in {current_year}?",
                    "The future of {topic} is being shaped by...",
                    "Recent developments in {topic} suggest...",
                    "The {topic} landscape is changing rapidly..."
                ],
                'trend_indicators': [
                    "Industry experts predict...",
                    "Market analysis shows...",
                    "Early adopters are already...",
                    "Leading indicators suggest...",
                    "Current patterns indicate..."
                ],
                'future_predictions': [
                    "In the next few years, we can expect...",
                    "The trajectory suggests...",
                    "Looking ahead, {topic} will likely...",
                    "Future developments may include...",
                    "The next phase of evolution will bring..."
                ]
            },
            'myth_busting': {
                'intro_templates': [
                    "Common myths about {topic} debunked",
                    "The truth behind {topic} misconceptions",
                    "Separating fact from fiction in {topic}",
                    "Let's clear up some confusion about {topic}",
                    "Time to bust some myths about {topic}"
                ],
                'myth_formats': [
                    "Myth #{num}: {myth}",
                    "Common Belief: {myth}",
                    "You might have heard: {myth}",
                    "Popular Misconception: {myth}",
                    "False Claim: {myth}"
                ],
                'fact_formats': [
                    "Fact: {fact}",
                    "Reality: {fact}",
                    "Truth: {fact}",
                    "Actually: {fact}",
                    "Here's the truth: {fact}"
                ]
            }
        }
        
        self.engagement_elements = {
            'questions': [
                "What's your experience with {topic}?",
                "Have you ever encountered {situation}?",
                "How would you handle {scenario}?",
                "What do you think about {topic}?",
                "Have you tried this approach before?",
                "What's your take on this?",
                "Can you relate to this situation?",
                "What would you do differently?"
            ],
            'calls_to_action': [
                "Share your thoughts in the comments below!",
                "Try this approach and let us know how it works for you!",
                "Join the discussion and share your experiences!",
                "Subscribe for more insights on {topic}!",
                "Follow us for daily tips on {topic}!",
                "Download our free guide on {topic}!",
                "Sign up for our newsletter!",
                "Connect with us on social media!"
            ],
            'interactive_elements': [
                "📊 Poll: What's your preferred approach to {topic}?",
                "🤔 Quick Quiz: Test your knowledge of {topic}",
                "💡 Challenge: Try implementing these tips this week",
                "✍️ Worksheet: Download our {topic} planning template",
                "🎯 Goal Setting: What's your target for {topic}?",
                "📝 Checklist: Rate your current {topic} strategy",
                "🔍 Self-Assessment: How well do you know {topic}?",
                "🎮 Interactive Demo: Try our {topic} simulator"
            ],
            'emotional_hooks': [
                "Imagine the possibilities...",
                "You won't believe what happens next...",
                "Here's the secret to success...",
                "This changed everything for me...",
                "The surprising truth about...",
                "What nobody tells you about...",
                "The hidden benefits of...",
                "Why you can't ignore..."
            ]
        }

    def get_style(self, style_name: str) -> Dict:
        """Get a specific writing style configuration."""
        return self.styles.get(style_name, self.styles['storytelling'])

    def get_random_style(self) -> Tuple[str, Dict]:
        """Get a random writing style."""
        style_name = random.choice(list(self.styles.keys()))
        return style_name, self.styles[style_name]

    def get_engagement_element(self, element_type: str, **kwargs) -> str:
        """Get a random engagement element with formatting."""
        try:
            elements = self.engagement_elements.get(element_type, [])
            if elements:
                element = random.choice(elements)
                return element.format(**kwargs)
            return ""
        except Exception as e:
            logger.error(f"Error getting engagement element: {e}")
            return ""

    def get_style_elements(self, style_name: str, section: str) -> List[str]:
        """Get style-specific elements for a section."""
        try:
            style = self.get_style(style_name)
            if section in style:
                return style[section]
            return []
        except Exception as e:
            logger.error(f"Error getting style elements: {e}")
            return []

    def enhance_content(self, content: str, style_name: str) -> str:
        """Enhance content with style-specific elements."""
        try:
            style = self.get_style(style_name)
            enhanced_content = content

            # Add transitions
            if 'transitions' in style:
                sentences = content.split('. ')
                for i in range(len(sentences) // 3):
                    idx = i * 3
                    transition = random.choice(style['transitions'])
                    sentences.insert(idx, transition)
                enhanced_content = '. '.join(sentences)

            # Add style-specific formatting
            if style_name == 'tutorial':
                enhanced_content = self._format_tutorial(enhanced_content)
            elif style_name == 'myth_busting':
                enhanced_content = self._format_myth_busting(enhanced_content)

            return enhanced_content

        except Exception as e:
            logger.error(f"Error enhancing content: {e}")
            return content

    def _format_tutorial(self, content: str) -> str:
        """Format content as a tutorial."""
        lines = content.split('\n')
        formatted_lines = []
        step_num = 1

        for line in lines:
            if line.lower().startswith(('step', 'part', 'phase')):
                formatted_lines.append(f"Step {step_num}: {line.split(':', 1)[1] if ':' in line else line}")
                step_num += 1
            else:
                formatted_lines.append(line)

        return '\n'.join(formatted_lines)

    def _format_myth_busting(self, content: str) -> str:
        """Format content for myth busting."""
        lines = content.split('\n')
        formatted_lines = []
        myth_num = 1

        for line in lines:
            if line.lower().startswith('myth'):
                formatted_lines.append(f"Myth #{myth_num}: {line.split(':', 1)[1] if ':' in line else line}")
                myth_num += 1
            elif line.lower().startswith('fact'):
                formatted_lines.append(f"✓ Fact: {line.split(':', 1)[1] if ':' in line else line}")
            else:
                formatted_lines.append(line)

        return '\n'.join(formatted_lines)
