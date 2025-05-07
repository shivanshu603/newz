from typing import List, Dict, Tuple, Optional, Set, Any
import random
import re
import spacy
import json
import os
import nltk
from nltk.corpus import wordnet as wn
from pathlib import Path
from utils import logger
from collections import defaultdict

class ContentHumanizer:
    def __init__(self):
        """Initialize the humanizer with advanced humanization capabilities"""
        try:
            # Load spaCy model for NLP processing
            self.nlp = spacy.load('en_core_web_sm')
            
            # Load resources for humanization
            self.idioms = self._load_idioms()
            self.transition_phrases = self._load_transitions()
            self.colloquialisms = self._load_colloquialisms()
            self.sensory_phrases = self._load_sensory_phrases()
            self.emotional_phrases = self._load_emotional_phrases()
            self.redundancy_patterns = self._load_redundancy_patterns()
            self.synonym_map = self._load_synonym_map()
            
            # Humanization settings
            self.burstiness_factor = 0.7  # Controls sentence length variation (0-1)
            self.perplexity_factor = 0.65  # Controls word choice randomness (0-1)
            self.error_rate = 0.03  # Rate of intentional minor "errors" (0-1)
            self.personal_touch_rate = 0.15  # Rate of adding personal elements (0-1)
            
            logger.info("ContentHumanizer initialized successfully with advanced features")
        except Exception as e:
            logger.error(f"Error initializing humanizer: {e}")
            raise

    def filter_unwanted_tokens(self, content: str) -> str:
        """
        Filter out emojis, special characters, unwanted commas, and other unwanted tokens
        using spaCy token analysis.
        """
        try:
            # First, remove emojis using regex
            emoji_pattern = re.compile(
                "["
                "\U0001F600-\U0001F64F"  # emoticons
                "\U0001F300-\U0001F5FF"  # symbols & pictographs
                "\U0001F680-\U0001F6FF"  # transport & map symbols
                "\U0001F700-\U0001F77F"  # alchemical symbols
                "\U0001F780-\U0001F7FF"  # Geometric Shapes
                "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                "\U0001FA00-\U0001FA6F"  # Chess Symbols
                "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                "\U00002702-\U000027B0"  # Dingbats
                "\U000024C2-\U0001F251"
                "]+", flags=re.UNICODE)
            content = emoji_pattern.sub(r'', content)
            
            # Remove other common problematic characters
            content = re.sub(r'[^\x00-\x7F]+', '', content)  # Remove non-ASCII characters
            content = re.sub(r'[^\w\s.,;:!?\'\"()-]', '', content)  # Keep only alphanumeric and basic punctuation
            
            # Process with spaCy for more advanced filtering
            doc = self.nlp(content)
            filtered_tokens = []
            
            for token in doc:
                # Skip specific token types
                if token.is_space:
                    filtered_tokens.append(token.text)
                    continue
                    
                if token.like_email or token.like_url:
                    continue
                    
                if token.is_punct:
                    # Allow only basic punctuation
                    if token.text in {',', '.', ';', ':', '!', '?', '-', '(', ')', '"', "'"}:
                        filtered_tokens.append(token.text)
                    continue
                    
                # Skip symbols and special characters
                if token.pos_ == 'SYM':
                    continue
                    
                # Skip tokens that are only special characters
                if all(not c.isalnum() and not c.isspace() for c in token.text):
                    continue
                    
                # Add valid tokens
                filtered_tokens.append(token.text_with_ws)
                
            # Join tokens and clean up spacing
            filtered_text = ''.join(filtered_tokens)
            
            # Fix common formatting issues
            filtered_text = re.sub(r'\s+', ' ', filtered_text)  # Remove multiple spaces
            filtered_text = re.sub(r'\s+([.,;:!?])', r'\1', filtered_text)  # Fix punctuation spacing
            filtered_text = re.sub(r'([.,;:!?])\s+([.,;:!?])', r'\1\2', filtered_text)  # Fix consecutive punctuation
            filtered_text = re.sub(r'\.{2,}', '...', filtered_text)  # Standardize ellipses
            filtered_text = re.sub(r',,+', ',', filtered_text)  # Remove multiple commas
            filtered_text = re.sub(r'::+', ':', filtered_text)  # Remove multiple colons
            filtered_text = re.sub(r';;+', ';', filtered_text)  # Remove multiple semicolons
            
            # Remove any remaining problematic sequences
            filtered_text = re.sub(r'[^\w\s.,;:!?\'\"()-]', '', filtered_text)
            
            return filtered_text.strip()
        except Exception as e:
            logger.error(f"Error filtering unwanted tokens: {e}")
            return content

    def humanize(self, content: str, topic: str = None) -> str:
        """
        Apply all humanization techniques to make content appear more human-written
        while preserving heading tags and structure
        
        Args:
            content: The AI-generated content to humanize
            topic: Optional topic to guide contextual humanization
            
        Returns:
            Humanized content that appears more natural and human-written
        """
        try:
            # Extract and preserve headings before processing
            heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
            headings = [(m.group(0), m.start(), m.end()) for m in heading_pattern.finditer(content)]
            
            # Replace headings with placeholders to protect them during processing
            placeholder_map = {}
            protected_content = content
            
            for i, (heading, start, end) in enumerate(headings):
                placeholder = f"__HEADING_PLACEHOLDER_{i}__"
                placeholder_map[placeholder] = heading
                protected_content = protected_content.replace(heading, placeholder)
            
            # Filter unwanted tokens first (excluding placeholders)
            filtered_content = self.filter_unwanted_tokens(protected_content)
            
            # Restore placeholders that might have been affected by filtering
            for placeholder, heading in placeholder_map.items():
                if placeholder not in filtered_content:
                    filtered_content = re.sub(r'__HEADING_PLACEHOLDER_\d+__', placeholder, filtered_content, count=1)
            
            # Initial analysis of content
            content_stats = self._analyze_content(filtered_content)
            logger.info(f"Content analysis: {len(content_stats['sentences'])} sentences, " 
                       f"avg length: {content_stats['avg_sentence_length']:.1f} words")
            
            # Split by paragraphs but preserve heading placeholders
            paragraphs = []
            current_paragraph = ""
            
            for line in filtered_content.split('\n'):
                if any(placeholder in line for placeholder in placeholder_map.keys()):
                    # This is a heading placeholder line
                    if current_paragraph:
                        paragraphs.append(current_paragraph)
                        current_paragraph = ""
                    paragraphs.append(line)
                elif not line.strip():
                    # Empty line - paragraph break
                    if current_paragraph:
                        paragraphs.append(current_paragraph)
                        current_paragraph = ""
                else:
                    # Regular content line
                    if current_paragraph:
                        current_paragraph += "\n" + line
                    else:
                        current_paragraph = line
            
            # Add the last paragraph if there is one
            if current_paragraph:
                paragraphs.append(current_paragraph)
            
            # Track overall document state for coherence
            doc_state = {
                "introduced_idioms": set(),
                "used_transitions": set(),
                "paragraph_count": len(paragraphs),
                "current_paragraph": 0,
                "topic": topic
            }
            
            humanized = []
            
            for i, para in enumerate(paragraphs):
                doc_state["current_paragraph"] = i
                
                # Skip humanization for heading placeholders
                if any(placeholder in para for placeholder in placeholder_map.keys()):
                    humanized.append(para)
                    continue
                
                # Apply multiple humanization techniques in sequence
                processed = para
                processed = self._vary_sentence_structure(processed)
                processed = self._add_natural_elements(processed, doc_state)
                processed = self._adjust_formality(processed)
                processed = self._add_transitions(processed, i, len(paragraphs))
                processed = self._add_burstiness(processed)
                processed = self._inject_colloquialisms(processed, doc_state)
                processed = self._add_personal_touch(processed, i, doc_state)
                processed = self._add_sensory_details(processed)
                processed = self._add_minor_imperfections(processed)
                
                humanized.append(processed)
            
            # Join paragraphs with proper spacing
            result = '\n\n'.join(humanized)
            
            # Apply overall coherence and flow (avoiding heading placeholders)
            result = self._ensure_logical_flow(result)
            
            # Add some final human-like touches
            if random.random() < 0.3:  # 30% chance
                result = self._add_concluding_thought(result)
            
            # Restore heading placeholders with actual headings
            for placeholder, heading in placeholder_map.items():
                result = result.replace(placeholder, heading)
            
            logger.info(f"Content successfully humanized with preserved headings: {len(result)} chars")
            return result
        except Exception as e:
            logger.error(f"Error humanizing content: {e}")
            # Return original content if humanization fails
            return content


    def _analyze_content(self, content: str) -> Dict[str, Any]:
        """Analyze content to guide humanization strategy"""
        try:
            doc = self.nlp(content)
            sentences = list(doc.sents)
            
            # Calculate basic statistics
            sentence_lengths = [len(sent.text.split()) for sent in sentences]
            avg_sentence_length = sum(sentence_lengths) / max(len(sentence_lengths), 1)
            
            # Detect formality level
            formality_indicators = {
                'formal': ['therefore', 'consequently', 'furthermore', 'thus', 'hence'],
                'academic': ['research', 'study', 'analysis', 'conclude', 'evidence'],
                'casual': ['really', 'basically', 'actually', 'pretty', 'kind of', 'sort of']
            }
            
            formality_scores = {category: 0 for category in formality_indicators}
            for category, words in formality_indicators.items():
                for word in words:
                    if re.search(r'\b' + word + r'\b', content.lower()):
                        formality_scores[category] += 1
            
            dominant_style = max(formality_scores.items(), key=lambda x: x[1])[0]
            if sum(formality_scores.values()) == 0:
                dominant_style = 'neutral'
                
            return {
                'sentences': sentences,
                'sentence_lengths': sentence_lengths,
                'avg_sentence_length': avg_sentence_length,
                'dominant_style': dominant_style,
                'word_count': len(content.split()),
                'paragraph_count': content.count('\n\n') + 1
            }
        except Exception as e:
            logger.error(f"Error analyzing content: {e}")
            return {
                'sentences': [],
                'sentence_lengths': [],
                'avg_sentence_length': 15,
                'dominant_style': 'neutral',
                'word_count': len(content.split()),
                'paragraph_count': content.count('\n\n') + 1
            }

    def _vary_sentence_structure(self, text: str) -> str:
        """Vary sentence structure and complexity for more human-like patterns"""
        try:
            doc = self.nlp(text)
            varied = []
            
            for i, sent in enumerate(doc.sents):
                # Apply different transformations based on position and randomness
                if i % 4 == 0 and len(sent.text.split()) > 5:  # Every 4th sentence if long enough
                    varied.append(self._make_complex(sent.text))
                elif i % 5 == 1:  # Every 5th sentence (offset by 1)
                    varied.append(self._make_simple(sent.text))
                elif i % 7 == 3 and random.random() < 0.7:  # Add some randomness
                    varied.append(self._add_emphasis(sent.text))
                else:
                    # Sometimes leave as is for natural variation
                    varied.append(sent.text)
            
            return ' '.join(varied)
        except Exception as e:
            logger.error(f"Error varying sentence structure: {e}")
            return text

    def _make_complex(self, sentence: str) -> str:
        """Make a sentence more complex and sophisticated with emotional depth"""
        try:
            # Don't modify very short sentences
            if len(sentence.split()) < 4:
                return sentence
                
            # Add subordinate clauses or descriptive phrases
            doc = self.nlp(sentence)
            
            # Find the subject and verb for potential modification
            subject = None
            verb = None
            
            for token in doc:
                if token.dep_ in ('nsubj', 'nsubjpass') and not subject:
                    subject = token
                if token.pos_ == 'VERB' and not verb:
                    verb = token
            
            if subject and random.random() < 0.7:
                # Add a descriptive phrase to the subject
                descriptive_phrases = [
                    f", {random.choice(['clearly', 'evidently', 'obviously', 'undoubtedly'])} {random.choice(['focused', 'determined', 'committed', 'dedicated'])}, ",
                    f", {random.choice(['with a sense of', 'showing', 'demonstrating', 'exhibiting'])} {random.choice(['purpose', 'determination', 'resolve', 'conviction'])}, ",
                    f", {random.choice(['despite', 'notwithstanding', 'regardless of'])} {random.choice(['the challenges', 'the difficulties', 'the obstacles', 'the hurdles'])}, "
                ]
                
                # Add emotional phrases 30% of the time
                if self.emotional_phrases and random.random() < 0.3:
                    emotional_phrase = random.choice(self.emotional_phrases)
                    descriptive_phrases.append(f", {emotional_phrase}, ")
                
                phrase = random.choice(descriptive_phrases)
                sentence = sentence.replace(subject.text, subject.text + phrase, 1)
            
            if verb and random.random() < 0.6:
                # Add an adverbial phrase
                adverbial_phrases = [
                    f" {random.choice(['quite', 'rather', 'somewhat', 'decidedly'])} {random.choice(['effectively', 'efficiently', 'successfully', 'impressively'])}",
                    f" {random.choice(['with great', 'with considerable', 'with remarkable', 'with notable'])} {random.choice(['skill', 'precision', 'expertise', 'finesse'])}",
                    f" {random.choice(['in a manner that', 'in a way that', 'such that it'])} {random.choice(['impressed', 'surprised', 'amazed', 'astonished'])} {random.choice(['observers', 'onlookers', 'viewers', 'witnesses'])}"
                ]
                
                # Add emotional phrases 30% of the time
                if self.emotional_phrases and random.random() < 0.3:
                    emotional_phrase = random.choice(self.emotional_phrases)
                    adverbial_phrases.append(f" {emotional_phrase}")
                
                phrase = random.choice(adverbial_phrases)
                sentence = sentence.replace(verb.text, verb.text + phrase, 1)
            
            # Sometimes add a dependent clause at the beginning
            if random.random() < 0.3:
                dependent_clauses = [
                    f"As one might expect, ",
                    f"Interestingly enough, ",
                    f"Given the circumstances, ",
                    f"When considering all factors, ",
                    f"While it may seem surprising, "
                ]
                
                # Add emotional phrases 20% of the time
                if self.emotional_phrases and random.random() < 0.2:
                    emotional_phrase = random.choice(self.emotional_phrases)
                    dependent_clauses.append(f"{emotional_phrase}, ")
                
                sentence = random.choice(dependent_clauses) + sentence[0].lower() + sentence[1:]
            
            return sentence
        except Exception as e:
            logger.error(f"Error making sentence complex: {e}")
            return sentence

    def _make_simple(self, sentence: str) -> str:
        """Simplify a sentence for variety"""
        try:
            # Don't modify very short sentences
            if len(sentence.split()) < 6:
                return sentence
                
            # Remove adverbs and simplify structure
            doc = self.nlp(sentence)
            
            # Identify tokens to potentially remove
            removable_tokens = []
            for token in doc:
                # Look for adverbs and adjectives that aren't essential
                if (token.pos_ == 'ADV' and token.dep_ not in ('neg', 'advmod')) or \
                   (token.pos_ == 'ADJ' and token.dep_ not in ('acomp', 'amod')):
                    removable_tokens.append(token)
            
            # Remove some tokens randomly
            simplified = sentence
            for token in removable_tokens:
                if random.random() < 0.6:  # 60% chance to remove
                    simplified = simplified.replace(' ' + token.text + ' ', ' ')
            
            # Sometimes break into two sentences for simplicity
            if len(simplified.split()) > 12 and ',' in simplified and random.random() < 0.4:
                parts = simplified.split(',', 1)
                if len(parts) == 2 and len(parts[0].split()) > 3 and len(parts[1].split()) > 3:
                    simplified = parts[0] + '. ' + parts[1][1].upper() + parts[1][2:]
            
            return simplified
        except Exception as e:
            logger.error(f"Error making sentence simple: {e}")
            return sentence

    def _add_emphasis(self, sentence: str) -> str:
        """Add emphasis or expressiveness to a sentence"""
        try:
            emphasis_patterns = [
                (r'\b(important|significant|crucial|essential)\b', r'truly \1'),
                (r'\b(good|great|excellent)\b', r'remarkably \1'),
                (r'\b(bad|terrible|awful)\b', r'absolutely \1'),
                (r'\b(interesting|fascinating)\b', r'genuinely \1')
            ]
            
            emphasized = sentence
            for pattern, replacement in emphasis_patterns:
                if re.search(pattern, emphasized, re.IGNORECASE) and random.random() < 0.7:
                    emphasized = re.sub(pattern, replacement, emphasized, flags=re.IGNORECASE)
            
            # Sometimes add an emphatic phrase at the end
            if random.random() < 0.3 and not emphasized.endswith(('!', '?')):
                emphatic_endings = [
                    ", without a doubt",
                    ", indeed",
                    ", to say the least",
                    ", by all accounts"
                ]
                emphasized = emphasized[:-1] + random.choice(emphatic_endings) + emphasized[-1]
            
            return emphasized
        except Exception as e:
            logger.error(f"Error adding emphasis: {e}")
            return sentence

    def _add_natural_elements(self, text: str, doc_state: Dict) -> str:
        """Add natural human elements like hedging, opinions, emotional phrases, and personal touches"""
        try:
            doc = self.nlp(text)
            sentences = list(doc.sents)
            
            if not sentences:
                return text
                
            # Select 1-2 sentences to modify with natural elements
            num_to_modify = min(len(sentences), random.randint(1, 2))
            indices_to_modify = random.sample(range(len(sentences)), num_to_modify)
            
            modified_sentences = []
            for i, sent in enumerate(sentences):
                if i in indices_to_modify:
                    # Choose a natural element to add
                    element_type = random.choice(['hedge', 'opinion', 'question', 'emphasis', 'emotion'])
                    
                    if element_type == 'hedge' and len(sent.text.split()) > 5:
                        # Add hedging language
                        hedges = [
                            "It seems that ", 
                            "It appears that ", 
                            "Perhaps ", 
                            "It could be argued that ", 
                            "From what we can tell, "
                        ]
                        modified = random.choice(hedges) + sent.text[0].lower() + sent.text[1:]
                        
                    elif element_type == 'opinion' and len(sent.text.split()) > 4:
                        # Add opinion marker
                        opinions = [
                            "Interestingly, ", 
                            "Notably, ", 
                            "Surprisingly, ", 
                            "Remarkably, "
                        ]
                        modified = random.choice(opinions) + sent.text
                        
                    elif element_type == 'question' and i > len(sentences) // 2:
                        # Convert to rhetorical question
                        modified = sent.text[:-1] + "?"
                        if random.random() < 0.5:
                            modified += " It's worth considering."
                            
                    elif element_type == 'emotion' and self.emotional_phrases:
                        # Add emotional phrase
                        phrase = random.choice(self.emotional_phrases)
                        if sent.text.endswith('.'):
                            modified = f"{sent.text[:-1]}, {phrase}."
                        else:
                            modified = f"{sent.text}, {phrase}"
                            
                    else:  # emphasis
                        # Add emphasis
                        modified = self._add_emphasis(sent.text)
                        
                    modified_sentences.append(modified)
                else:
                    modified_sentences.append(sent.text)
            
            return ' '.join(modified_sentences)
        except Exception as e:
            logger.error(f"Error adding natural elements: {e}")
            return text

    def _adjust_formality(self, text: str) -> str:
        """Adjust formality level to make content more natural"""
        try:
            # Analyze current formality
            formality_score = self._calculate_formality(text)
            
            # If already balanced (0.4-0.6), don't modify
            if 0.4 <= formality_score <= 0.6:
                return text
                
            doc = self.nlp(text)
            sentences = list(doc.sents)
            
            modified_sentences = []
            for sent in sentences:
                if formality_score > 0.6:  # Too formal, make more casual
                    modified = self._make_more_casual(sent.text)
                else:  # Too casual, make more formal
                    modified = self._make_more_formal(sent.text)
                modified_sentences.append(modified)
            
            return ' '.join(modified_sentences)
        except Exception as e:
            logger.error(f"Error adjusting formality: {e}")
            return text

    def _calculate_formality(self, text: str) -> float:
        """Calculate formality score (0-1) where 1 is very formal"""
        try:
            formal_indicators = [
                r'\b(therefore|consequently|furthermore|thus|hence)\b',
                r'\b(demonstrate|illustrate|indicate|exhibit)\b',
                r'\b(additionally|moreover|subsequently)\b',
                r'\b(utilize|implement|facilitate)\b',
                r'\b(regarding|concerning|pertaining to)\b'
            ]
            
            casual_indicators = [
                r'\b(really|basically|actually|pretty much|kind of)\b',
                r'\b(stuff|things|guy|guys)\b',
                r'\b(awesome|cool|great|nice)\b',
                r'\b(like|you know|I mean)\b',
                r'\b(a lot|lots of|tons of)\b'
            ]
            
            formal_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in formal_indicators)
            casual_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in casual_indicators)
            
            total = formal_count + casual_count
            if total == 0:
                return 0.5  # Neutral if no indicators found
                
            return formal_count / total
        except Exception as e:
            logger.error(f"Error calculating formality: {e}")
            return 0.5

    def _make_more_casual(self, text: str) -> str:
        """Make text more casual and conversational"""
        try:
            # Replace formal phrases with casual alternatives
            replacements = [
                (r'\b(therefore)\b', 'so'),
                (r'\b(subsequently)\b', 'after that'),
                (r'\b(nevertheless)\b', 'still'),
                (r'\b(furthermore)\b', 'also'),
                (r'\b(utilize)\b', 'use'),
                (r'\b(implement)\b', 'put in place'),
                (r'\b(demonstrate)\b', 'show'),
                (r'\b(sufficient)\b', 'enough'),
                (r'\b(numerous)\b', 'many'),
                (r'\b(commence)\b', 'start'),
                (r'\b(terminate)\b', 'end'),
                (r'\b(endeavor)\b', 'try')
            ]
            
            casual = text
            for pattern, replacement in replacements:
                if re.search(pattern, casual, re.IGNORECASE) and random.random() < 0.7:
                    casual = re.sub(pattern, replacement, casual, flags=re.IGNORECASE)
            
            # Add casual markers occasionally
            if random.random() < 0.3 and len(casual) > 20:
                casual_markers = [
                    "You know, ",
                    "Honestly, ",
                    "Look, ",
                    "I think "
                ]
                casual = random.choice(casual_markers) + casual[0].lower() + casual[1:]
            
            return casual
        except Exception as e:
            logger.error(f"Error making text more casual: {e}")
            return text

    def _make_more_formal(self, text: str) -> str:
        """Make text more formal and professional"""
        try:
            # Replace casual phrases with formal alternatives
            replacements = [
                (r'\b(a lot of|lots of)\b', 'numerous'),
                (r'\b(get)\b', 'obtain'),
                (r'\b(show)\b', 'demonstrate'),
                (r'\b(but)\b', 'however'),
                (r'\b(use)\b', 'utilize'),
                (r'\b(start)\b', 'commence'),
                (r'\b(end)\b', 'conclude'),
                (r'\b(try)\b', 'attempt'),
                (r'\b(like)\b', 'such as'),
                (r'\b(big)\b', 'substantial'),
                (r'\b(small)\b', 'minimal')
            ]
            
            formal = text
            for pattern, replacement in replacements:
                if re.search(pattern, formal, re.IGNORECASE) and random.random() < 0.7:
                    formal = re.sub(pattern, replacement, formal, flags=re.IGNORECASE)
            
            # Remove very casual markers
            casual_markers = [
                r'\byou know\b',
                r'\bkind of\b',
                r'\bsort of\b',
                r'\bbasically\b',
                r'\bliterally\b'
            ]
            
            for marker in casual_markers:
                formal = re.sub(marker, '', formal, flags=re.IGNORECASE)
            
            return formal
        except Exception as e:
            logger.error(f"Error making text more formal: {e}")
            return text

    def _add_transitions(self, text: str, para_index: int, total_paras: int) -> str:
        """Add appropriate transition phrases based on paragraph position"""
        try:
            # Don't add transitions to very short paragraphs
            if len(text.split()) < 15:
                return text
                
            # Select transition type based on paragraph position
            if para_index == 0:
                # Introduction transitions
                transitions = [
                    "To begin with, ",
                    "First and foremost, ",
                    "At the outset, it's important to note that ",
                    "When considering this topic, ",
                    "Before diving deeper, "
                ]
            elif para_index == total_paras - 1:
                # Conclusion transitions
                transitions = [
                    "In conclusion, ",
                    "To sum up, ",
                    "Finally, ",
                    "All things considered, ",
                    "Taking everything into account, "
                ]
            else:
                # Middle paragraph transitions
                if para_index < total_paras // 2:
                    # Early-middle transitions
                    transitions = [
                        "Furthermore, ",
                        "Building on this idea, ",
                        "In addition to this, ",
                        "Following this line of thought, ",
                        "It's also worth noting that "
                    ]
                else:
                    # Late-middle transitions
                    transitions = [
                        "Moreover, ",
                        "Another important aspect is that ",
                        "Equally important is the fact that ",
                        "This leads us to consider that ",
                        "In the same vein, "
                    ]
            
            # Apply transition with some randomness
            if random.random() < 0.8:  # 80% chance to add transition
                transition = random.choice(transitions)
                # Make sure first letter after transition is lowercase
                text = transition + text[0].lower() + text[1:]
            
            return text
        except Exception as e:
            logger.error(f"Error adding transitions: {e}")
            return text

    def _add_burstiness(self, text: str) -> str:
        """Add burstiness by varying sentence length and structure"""
        try:
            doc = self.nlp(text)
            sentences = list(doc.sents)
            
            if len(sentences) < 3:
                return text
                
            # Calculate current sentence lengths
            lengths = [len(sent.text.split()) for sent in sentences]
            avg_length = sum(lengths) / len(lengths)
            
            # Decide if we need more variation
            if max(lengths) - min(lengths) < avg_length * 0.5:
                # Not enough variation, add more burstiness
                modified_sentences = []
                
                for i, sent in enumerate(sentences):
                    if i % 3 == 0 and len(sent.text.split()) > 5:
                        # Make this sentence shorter
                        modified = self._make_simple(sent.text)
                        # Sometimes make it very short for dramatic effect
                        if random.random() < 0.3:
                            words = modified.split()
                            if len(words) > 5:
                                cutoff = random.randint(3, min(5, len(words) - 1))
                                modified = ' '.join(words[:cutoff]) + '.'
                    elif i % 3 == 1:
                        # Keep this sentence as is
                        modified = sent.text
                    else:
                        # Make this sentence longer and more complex
                        modified = self._make_complex(sent.text)
                        
                    modified_sentences.append(modified)
                
                return ' '.join(modified_sentences)
            
            return text
        except Exception as e:
            logger.error(f"Error adding burstiness: {e}")
            return text

    def _inject_colloquialisms(self, text: str, doc_state: Dict) -> str:
        """Inject colloquialisms and idioms for natural human feel"""
        try:
            # Don't overdo colloquialisms
            if random.random() > self.perplexity_factor:
                return text
                
            doc = self.nlp(text)
            sentences = list(doc.sents)
            
            if not sentences:
                return text
                
            # Select a sentence to modify (prefer middle sentences)
            valid_indices = list(range(len(sentences)))
            if len(valid_indices) > 2:
                valid_indices = valid_indices[1:-1]  # Exclude first and last if possible
                
            if not valid_indices:
                return text
                
            idx = random.choice(valid_indices)
            
            # Choose what to inject
            if random.random() < 0.6:  # 60% chance for idiom
                # Select an idiom that hasn't been used yet if possible
                available_idioms = [i for i in self.idioms if i not in doc_state["introduced_idioms"]]
                if not available_idioms and self.idioms:
                    available_idioms = self.idioms
                    
                if available_idioms:
                    idiom = random.choice(available_idioms)
                    doc_state["introduced_idioms"].add(idiom)
                    
                    # Insert the idiom
                    sent_text = sentences[idx].text
                    if random.random() < 0.5 and not sent_text.startswith(("However", "Therefore", "Furthermore")):
                        # Add at beginning
                        modified = f"{idiom}, {sent_text[0].lower()}{sent_text[1:]}"
                    else:
                        # Add at end
                        if sent_text.endswith('.'):
                            modified = f"{sent_text[:-1]}, {idiom}."
                        else:
                            modified = f"{sent_text}, {idiom}"
                    
                    # Replace the original sentence
                    result = []
                    for i, sent in enumerate(sentences):
                        if i == idx:
                            result.append(modified)
                        else:
                            result.append(sent.text)
                    
                    return ' '.join(result)
            else:  # 40% chance for colloquialism
                # Select a colloquialism
                if self.colloquialisms:
                    colloquialism = random.choice(self.colloquialisms)
                    
                    # Find a suitable place to insert it
                    sent_text = sentences[idx].text
                    sent_doc = self.nlp(sent_text)
                    
                    # Look for verbs or adjectives to replace
                    for token in sent_doc:
                        if token.pos_ in ('VERB', 'ADJ') and len(token.text) > 3 and random.random() < 0.7:
                            # Replace with colloquialism
                            modified = sent_text.replace(token.text, colloquialism, 1)
                            
                            # Replace the original sentence
                            result = []
                            for i, sent in enumerate(sentences):
                                if i == idx:
                                    result.append(modified)
                                else:
                                    result.append(sent.text)
                            
                            return ' '.join(result)
            
            return text
        except Exception as e:
            logger.error(f"Error injecting colloquialisms: {e}")
            return text

    def _add_personal_touch(self, text: str, para_index: int, doc_state: Dict) -> str:
        """Add personal touches like anecdotes or opinions with emotional depth"""
        try:
            # Only add personal touches occasionally
            if random.random() > self.personal_touch_rate:
                return text
                
            # More likely to add personal touches in the first or last paragraphs
            if para_index > 0 and para_index < doc_state["paragraph_count"] - 1 and random.random() < 0.7:
                return text
                
            # Generate a personal touch based on the topic
            topic = doc_state.get("topic", "")
            
            # Base personal touches
            personal_touches = [
                f"I've always found that {topic or 'this subject'} resonates with many people.",
                f"In my experience, {topic or 'this area'} continues to evolve in fascinating ways.",
                f"You might be surprised by how often {topic or 'this topic'} comes up in everyday conversations.",
                f"It's worth remembering that {topic or 'this subject'} affects us all in different ways.",
                f"I've noticed that perspectives on {topic or 'this'} have changed significantly over time."
            ]
            
            # Enhanced with emotional phrases if available
            if self.emotional_phrases and random.random() < 0.6:
                emotional_phrase = random.choice(self.emotional_phrases)
                personal_touches.extend([
                    f"{emotional_phrase}, I've found {topic or 'this'} to be particularly meaningful.",
                    f"{emotional_phrase}, {topic or 'this subject'} has shaped my perspective in unexpected ways.",
                    f"My journey with {topic or 'this'} has been {emotional_phrase} at times."
                ])
            
            # Add the personal touch
            if para_index == 0:  # First paragraph
                # Add at the end
                if text.endswith('.'):
                    return f"{text} {random.choice(personal_touches)}"
                else:
                    return f"{text}. {random.choice(personal_touches)}"
            else:  # Last or other paragraph
                # Add at the beginning
                return f"{random.choice(personal_touches)} {text}"
        except Exception as e:
            logger.error(f"Error adding personal touch: {e}")
            return text

    def _add_sensory_details(self, text: str) -> str:
        """Add sensory and emotional details to make content more vivid"""
        try:
            # Only add details occasionally
            if random.random() > 0.3 or (not self.sensory_phrases and not self.emotional_phrases):
                return text
                
            doc = self.nlp(text)
            sentences = list(doc.sents)
            
            if not sentences or len(sentences) < 2:
                return text
                
            # Select a sentence to enhance with details
            idx = random.randint(0, len(sentences) - 1)
            
            # Choose between sensory or emotional phrase (or both)
            use_sensory = self.sensory_phrases and random.random() < 0.7
            use_emotional = self.emotional_phrases and random.random() < 0.7
            
            # Build the enhancement phrase
            enhancement = []
            if use_sensory:
                enhancement.append(random.choice(self.sensory_phrases))
            if use_emotional:
                enhancement.append(random.choice(self.emotional_phrases))
            
            if not enhancement:
                return text
                
            enhancement_phrase = ', '.join(enhancement)
            
            # Add the detail
            sent_text = sentences[idx].text
            if sent_text.endswith('.'):
                modified = f"{sent_text[:-1]}, {enhancement_phrase}."
            else:
                modified = f"{sent_text}, {enhancement_phrase}"
            
            # Replace the original sentence
            result = []
            for i, sent in enumerate(sentences):
                if i == idx:
                    result.append(modified)
                else:
                    result.append(sent.text)
            
            return ' '.join(result)
        except Exception as e:
            logger.error(f"Error adding sensory details: {e}")
            return text

    def _add_minor_imperfections(self, text: str) -> str:
        """Add minor imperfections to make text appear more human"""
        try:
            # Only add imperfections occasionally
            if random.random() > self.error_rate:
                return text
                
            imperfection_type = random.choice(['double_space', 'fragment', 'comma_splice'])
            
            if imperfection_type == 'double_space':
                # Add a double space somewhere in the text
                words = text.split()
                if len(words) < 5:
                    return text
                    
                idx = random.randint(1, len(words) - 2)
                words[idx] = words[idx] + "  "  # Double space
                return ' '.join(words)
                
            elif imperfection_type == 'fragment':
                # Add a sentence fragment
                fragments = [
                    "Absolutely.",
                    "Not always.",
                    "Quite remarkable.",
                    "Especially nowadays.",
                    "Without a doubt.",
                    "For obvious reasons."
                ]
                
                # Insert at a random position
                sentences = text.split('.')
                if len(sentences) < 3:
                    return text
                    
                idx = random.randint(1, len(sentences) - 2)
                sentences[idx] = sentences[idx] + ". " + random.choice(fragments)
                return '.'.join(sentences)
                
            elif imperfection_type == 'comma_splice':
                # Create a comma splice (two independent clauses joined by a comma)
                doc = self.nlp(text)
                sentences = list(doc.sents)
                
                if len(sentences) < 2:
                    return text
                    
                # Select two consecutive sentences to join
                idx = random.randint(0, len(sentences) - 2)
                
                # Join with a comma instead of a period
                first_sent = sentences[idx].text
                second_sent = sentences[idx + 1].text
                
                if first_sent.endswith('.'):
                    first_sent = first_sent[:-1]
                    
                # Make sure second sentence starts with lowercase
                if second_sent and len(second_sent) > 1:
                    second_sent = second_sent[0].lower() + second_sent[1:]
                
                spliced = f"{first_sent}, {second_sent}"
                
                # Replace the original sentences
                result = []
                for i, sent in enumerate(sentences):
                    if i == idx:
                        result.append(spliced)
                    elif i == idx + 1:
                        continue  # Skip the second sentence as it's now part of the splice
                    else:
                        result.append(sent.text)
                
                return ' '.join(result)
            
            return text
        except Exception as e:
            logger.error(f"Error adding minor imperfections: {e}")
            return text

    def _ensure_logical_flow(self, text: str) -> str:
        """Ensure logical flow and coherence throughout the text while preserving headings"""
        try:
            # This is a complex task that would ideally involve deep semantic analysis
            # For now, we'll focus on adding cohesive devices between paragraphs
            
            # Check for heading placeholders
            heading_placeholder_pattern = re.compile(r'__HEADING_PLACEHOLDER_\d+__')
            has_placeholders = bool(heading_placeholder_pattern.search(text))
            
            paragraphs = text.split('\n\n')
            if len(paragraphs) < 2:
                return text
                
            enhanced_paragraphs = [paragraphs[0]]  # Keep first paragraph as is
            
            for i in range(1, len(paragraphs)):
                prev_para = paragraphs[i-1]
                curr_para = paragraphs[i]
                
                # Skip processing if current paragraph is a heading placeholder
                if has_placeholders and heading_placeholder_pattern.search(curr_para):
                    enhanced_paragraphs.append(curr_para)
                    continue
                
                # Skip processing if previous paragraph is a heading placeholder
                if has_placeholders and heading_placeholder_pattern.search(prev_para):
                    enhanced_paragraphs.append(curr_para)
                    continue
                
                # Skip processing if current paragraph starts with a markdown heading
                if re.match(r'^#{1,6}\s+', curr_para):
                    enhanced_paragraphs.append(curr_para)
                    continue
                
                # Extract key terms from previous paragraph
                prev_doc = self.nlp(prev_para)
                key_terms = []
                
                for token in prev_doc:
                    if token.pos_ in ('NOUN', 'PROPN') and not token.is_stop and len(token.text) > 3:
                        key_terms.append(token.text.lower())
                
                # Check if current paragraph references any key terms
                curr_doc = self.nlp(curr_para)
                has_reference = False
                
                for token in curr_doc:
                    if token.text.lower() in key_terms:
                        has_reference = True
                        break
                
                # If no reference found, add a cohesive device
                if not has_reference and random.random() < 0.7:
                    cohesive_devices = [
                        f"Regarding this {random.choice(['topic', 'matter', 'issue', 'subject'])}, ",
                        f"With this in mind, ",
                        f"Building on these {random.choice(['ideas', 'concepts', 'points', 'observations'])}, ",
                        f"Following this line of thought, ",
                        f"In light of these {random.choice(['considerations', 'factors', 'aspects', 'elements'])}, "
                    ]
                    
                    # Only modify if paragraph doesn't start with a heading
                    if not re.match(r'^#{1,6}\s+', curr_para):
                        curr_para = random.choice(cohesive_devices) + curr_para[0].lower() + curr_para[1:]
                
                enhanced_paragraphs.append(curr_para)
            
            return '\n\n'.join(enhanced_paragraphs)
        except Exception as e:
            logger.error(f"Error ensuring logical flow: {e}")
            return text

    def _add_concluding_thought(self, text: str) -> str:
        """Add a concluding thought or reflection at the very end"""
        try:
            concluding_thoughts = [
                "It's worth reflecting on how this continues to shape our understanding today.",
                "The implications of this extend far beyond what we might initially consider.",
                "As we move forward, these insights will undoubtedly prove valuable.",
                "This perspective offers a fresh way of looking at a familiar challenge.",
                "Ultimately, this reminds us of the complexity inherent in the subject."
            ]
            
            # Add a line break and the concluding thought
            return f"{text}\n\n{random.choice(concluding_thoughts)}"
        except Exception as e:
            logger.error(f"Error adding concluding thought: {e}")
            return text

    def _load_idioms(self) -> List[str]:
        """Load common idioms and expressions"""
        try:
            # Default idioms if file not found
            default_idioms = [
                "at the end of the day",
                "the best of both worlds",
                "speak of the devil",
                "see eye to eye",
                "once in a blue moon",
                "when pigs fly",
                "costs an arm and a leg",
                "break the ice",
                "hit the nail on the head",
                "under the weather",
                "piece of cake",
                "let the cat out of the bag",
                "feeling under the weather",
                "back to square one",
                "bite off more than you can chew",
                "a blessing in disguise",
                "call it a day",
                "cutting corners",
                "get your act together",
                "go back to the drawing board",
                "hang in there",
                "hit the books",
                "it's not rocket science",
                "miss the boat",
                "pull yourself together",
                "so far so good",
                "speak of the devil",
                "that's the last straw",
                "the ball is in your court",
                "the best of both worlds"
            ]
            
            # Try to load from file
            try:
                file_path = os.path.join(os.path.dirname(__file__), 'data', 'idioms.json')
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load idioms from file: {e}")
                
            return default_idioms
        except Exception as e:
            logger.error(f"Error loading idioms: {e}")
            return []

    def _load_transitions(self) -> Dict[str, List[str]]:
        """Load transition phrases for different paragraph positions"""
        try:
            # Default transitions if file not found
            default_transitions = {
                "introduction": [
                    "To begin with",
                    "First and foremost",
                    "At the outset",
                    "To start with",
                    "Initially"
                ],
                "middle": [
                    "Furthermore",
                    "In addition",
                    "Moreover",
                    "Additionally",
                    "Besides",
                    "What's more",
                    "Another key point",
                    "Equally important"
                ],
                "contrast": [
                    "However",
                    "On the other hand",
                    "In contrast",
                    "Nevertheless",
                    "Conversely",
                    "Despite this",
                    "Notwithstanding"
                ],
                "example": [
                    "For instance",
                    "For example",
                    "To illustrate",
                    "As an example",
                    "Specifically",
                    "To demonstrate"
                ],
                "conclusion": [
                    "In conclusion",
                    "To sum up",
                    "Finally",
                    "In summary",
                    "To conclude",
                    "All things considered",
                    "In the final analysis"
                ]
            }
            
            # Try to load from file
            try:
                file_path = os.path.join(os.path.dirname(__file__), 'data', 'transitions.json')
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load transitions from file: {e}")
                
            return default_transitions
        except Exception as e:
            logger.error(f"Error loading transitions: {e}")
            return {}

    def _load_colloquialisms(self) -> List[str]:
        """Load colloquial expressions"""
        try:
            # Default colloquialisms if file not found
            default_colloquialisms = [
                "hit the spot",
                "chill out",
                "hang out",
                "no biggie",
                "awesome",
                "cool beans",
                "rock solid",
                "bummed out",
                "hang in there",
                "no sweat",
                "ace it",
                "all set",
                "catch you later",
                "dig it",
                "epic fail",
                "feeling blue",
                "get it",
                "heads up",
                "in the zone",
                "keep tabs on",
                "legit",
                "my bad",
                "on point",
                "psyched",
                "sweet",
                "vibe",
                "what's up",
                "you bet",
                "zonked out"
            ]
            
            # Try to load from file
            try:
                file_path = os.path.join(os.path.dirname(__file__), 'data', 'colloquialisms.json')
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load colloquialisms from file: {e}")
                
            return default_colloquialisms
        except Exception as e:
            logger.error(f"Error loading colloquialisms: {e}")
            return []

    def _load_sensory_phrases(self) -> List[str]:
        """Load sensory phrases for vivid descriptions"""
        try:
            # Default sensory phrases if file not found
            default_sensory_phrases = [
                "filling the air with a subtle aroma",
                "creating a symphony of colors",
                "with a texture smooth as silk",
                "leaving a bitter aftertaste",
                "casting long shadows in the fading light",
                "with a deafening roar that echoed through the valley",
                "sending shivers down my spine",
                "with a warmth that spread through my fingers",
                "tasting of summer and childhood memories",
                "with a scent reminiscent of fresh rain",
                "feeling rough and weathered to the touch",
                "glowing with an ethereal blue light",
                "with a sweetness that lingered on the tongue",
                "creating a cacophony of sounds",
                "with a chill that cut to the bone"
            ]
            
            # Try to load from file
            try:
                file_path = os.path.join(os.path.dirname(__file__), 'data', 'sensory_phrases.json')
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load sensory phrases from file: {e}")
                
            return default_sensory_phrases
        except Exception as e:
            logger.error(f"Error loading sensory phrases: {e}")
            return []

    def _load_emotional_phrases(self) -> List[str]:
        """Load emotional phrases to add depth"""
        try:
            # Try to load from file
            try:
                file_path = os.path.join(os.path.dirname(__file__), 'data', 'emotional_phrases.json')
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        phrases = json.load(f)
                        if phrases:  # Only return if we got valid phrases
                            return phrases
            except Exception as e:
                logger.warning(f"Could not load emotional phrases from file: {e}")
                
            # Return empty list if file load fails - we'll use other techniques
            return []
        except Exception as e:
            logger.error(f"Error loading emotional phrases: {e}")
            return []

    def _load_redundancy_patterns(self) -> List[Tuple[str, str]]:
        """Load patterns for adding natural redundancies"""
        try:
            # Default redundancy patterns if file not found
            default_redundancy_patterns = [
                (r'\b(important)\b', r'really \1'),
                (r'\b(difficult)\b', r'quite \1'),
                (r'\b(interesting)\b', r'rather \1'),
                (r'\b(good)\b', r'pretty \1'),
                (r'\b(bad)\b', r'quite \1'),
                (r'\b(surprising)\b', r'somewhat \1'),
                (r'\b(large)\b', r'fairly \1'),
                (r'\b(small)\b', r'relatively \1'),
                (r'\b(new)\b', r'brand \1'),
                (r'\b(old)\b', r'quite \1')
            ]
            
            # Try to load from file
            try:
                file_path = os.path.join(os.path.dirname(__file__), 'data', 'redundancy_patterns.json')
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        patterns = json.load(f)
                        return [(p[0], p[1]) for p in patterns]
            except Exception as e:
                logger.warning(f"Could not load redundancy patterns from file: {e}")
                
            return default_redundancy_patterns
        except Exception as e:
            logger.error(f"Error loading redundancy patterns: {e}")
            return []

    def _load_synonym_map(self) -> Dict[str, Dict[str, Any]]:
        """Load enhanced synonym map with contexts and frequencies"""
        try:
            # Default synonym map if file not found
            default_synonym_map = {
                "good": {
                    "synonyms": ["great", "excellent", "fine", "wonderful", "positive"],
                    "contexts": {
                        "general": ["good", "great", "positive"],
                        "performance": ["excellent", "outstanding", "superb"]
                    },
                    "frequencies": {
                        "great": 0.25,
                        "excellent": 0.15,
                        "fine": 0.1,
                        "wonderful": 0.1,
                        "positive": 0.1
                    }
                },
                "bad": {
                    "synonyms": ["poor", "terrible", "awful", "negative", "unpleasant"],
                    "contexts": {
                        "general": ["bad", "negative", "unpleasant"],
                        "performance": ["poor", "substandard", "inferior"]
                    },
                    "frequencies": {
                        "poor": 0.25,
                        "terrible": 0.2,
                        "awful": 0.15,
                        "negative": 0.1,
                        "unpleasant": 0.1
                    }
                }
            }
            
            # Try to load from file
            try:
                file_path = os.path.join(os.path.dirname(__file__), 'data', 'synonyms.json')
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load synonym map from file: {e}")
                
            return default_synonym_map
        except Exception as e:
            logger.error(f"Error loading synonym map: {e}")
            return {}
