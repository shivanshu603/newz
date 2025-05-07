import re
from typing import Dict, List
from utils import logger

class SEOValidator:
    OPTIMAL_RANGES = {
        'heading_keyword_ratio': (0.15, 0.25),  # 15-25% of headings should contain keywords
        'content_density': (0.5, 1.5),  # Keyword density percentage
        'semantic_variation': (0.7, 1.0),  # Semantic relevance score
        'question_coverage': (2, 5)  # Number of questions answered
    }

    def validate(self, content: str, keyword_data: Dict) -> Dict:
        """Return SEO health report with optimization suggestions"""
        report = {
            'primary_keyword_usage': self._check_primary_usage(content, keyword_data),
            'heading_distribution': self._analyze_headings(content),
            'semantic_cohesion': self._calculate_semantic_score(content, keyword_data),
            'faq_presence': self._count_questions_answered(content, keyword_data)
        }
        
        return self._generate_optimization_suggestions(report)

    def _check_primary_usage(self, content: str, keyword_data: Dict) -> Dict:
        """Analyze primary keyword usage and placement"""
        primary_keywords = keyword_data.get('primary', [])
        if not primary_keywords:
            return {
                'score': 0,
                'message': 'No primary keywords found',
                'details': {
                    'found_in_title': False,
                    'found_in_first_paragraph': False,
                    'found_in_headings': 0,
                    'total_occurrences': 0
                }
            }

        # Split content into sections
        sections = content.split('\n\n')
        if not sections:
            return {
                'score': 0,
                'message': 'No content to analyze',
                'details': {
                    'found_in_title': False,
                    'found_in_first_paragraph': False,
                    'found_in_headings': 0,
                    'total_occurrences': 0
                }
            }

        # Initialize results
        results = {
            'found_in_title': False,
            'found_in_first_paragraph': False,
            'found_in_headings': 0,
            'total_occurrences': 0
        }

        # Check title (first h1)
        title_match = re.search(r'<h1>(.*?)</h1>', content, re.IGNORECASE)
        if title_match:
            title = title_match.group(1)
            results['found_in_title'] = any(kw.lower() in title.lower() for kw in primary_keywords)

        # Check first paragraph
        first_para = next((s for s in sections if not s.strip().startswith('<h')), '')
        results['found_in_first_paragraph'] = any(kw.lower() in first_para.lower() for kw in primary_keywords)

        # Check headings
        headings = re.findall(r'<h[1-6]>(.*?)</h[1-6]>', content, re.IGNORECASE)
        results['found_in_headings'] = sum(1 for h in headings if any(kw.lower() in h.lower() for kw in primary_keywords))

        # Count total occurrences
        for kw in primary_keywords:
            results['total_occurrences'] += len(re.findall(rf'\b{re.escape(kw)}\b', content, re.IGNORECASE))

        # Calculate score
        score = 0
        if results['found_in_title']: score += 0.3
        if results['found_in_first_paragraph']: score += 0.2
        if results['found_in_headings'] >= 2: score += 0.3
        if results['total_occurrences'] >= 3: score += 0.2

        return {
            'score': score,
            'details': results,
            'message': self._generate_primary_usage_message(results)
        }

    def _analyze_headings(self, content: str) -> Dict:
        """Analyze heading structure and distribution"""
        heading_counts = {f'h{i}': 0 for i in range(1, 7)}
        
        # Count headings
        for level in range(1, 7):
            heading_counts[f'h{level}'] = len(re.findall(
                rf'<h{level}>.*?</h{level}>', content, re.IGNORECASE
            ))

        # Check heading hierarchy
        hierarchy_valid = True
        prev_level = 0
        for match in re.finditer(r'<h(\d)>', content, re.IGNORECASE):
            level = int(match.group(1))
            if level - prev_level > 1:  # Skip heading level
                hierarchy_valid = False
                break
            prev_level = level

        return {
            'counts': heading_counts,
            'hierarchy_valid': hierarchy_valid,
            'total_headings': sum(heading_counts.values())
        }

    def _calculate_semantic_score(self, content: str, keyword_data: Dict) -> float:
        """Calculate semantic relevance score"""
        try:
            # Get all keyword variations
            all_keywords = set()
            for key in ['primary', 'secondary', 'long_tail']:
                all_keywords.update(keyword_data.get(key, []))
            
            if not all_keywords:
                return 0.0

            # Count matches including semantic variations
            matches = 0
            word_count = len(content.split())
            
            for keyword in all_keywords:
                # Count exact matches
                matches += len(re.findall(rf'\b{re.escape(keyword)}\b', content, re.IGNORECASE))
                
                # Count partial matches for compound keywords
                if ' ' in keyword:
                    parts = keyword.split()
                    for part in parts:
                        if len(part) > 3:  # Only count significant words
                            matches += len(re.findall(rf'\b{re.escape(part)}\b', content, re.IGNORECASE)) * 0.5

            return min(1.0, matches / (word_count * self.OPTIMAL_RANGES['content_density'][1]))

        except Exception as e:
            logger.error(f"Error calculating semantic score: {e}")
            return 0.0

    def _count_questions_answered(self, content: str, keyword_data: Dict) -> Dict:
        """Check how many target questions are answered in the content"""
        target_questions = keyword_data.get('questions', [])
        if not target_questions:
            return {'count': 0, 'score': 0, 'message': 'No target questions found'}

        answered_count = 0
        for question in target_questions:
            # Look for question or its key terms in content
            question_terms = set(re.findall(r'\b\w+\b', question.lower()))
            content_terms = set(re.findall(r'\b\w+\b', content.lower()))
            
            # Calculate term overlap
            overlap = len(question_terms & content_terms) / len(question_terms)
            if overlap >= 0.7:  # 70% term overlap threshold
                answered_count += 1

        score = min(1.0, answered_count / self.OPTIMAL_RANGES['question_coverage'][1])
        
        return {
            'count': answered_count,
            'score': score,
            'message': f"Answered {answered_count} out of {len(target_questions)} target questions"
        }

    def _generate_primary_usage_message(self, results: Dict) -> str:
        """Generate feedback message for primary keyword usage"""
        messages = []
        
        if not results['found_in_title']:
            messages.append("Add primary keyword to the title")
            
        if not results['found_in_first_paragraph']:
            messages.append("Include primary keyword in the first paragraph")
            
        if results['found_in_headings'] < 2:
            messages.append("Use primary keyword in more headings")
            
        if results['total_occurrences'] < 3:
            messages.append("Increase primary keyword usage throughout content")
            
        return " | ".join(messages) if messages else "Primary keyword usage is optimal"

    def _generate_optimization_suggestions(self, report: Dict) -> Dict:
        """Generate actionable optimization suggestions based on the report"""
        suggestions = []
        score = 0.0
        
        # Primary keyword optimization
        if report['primary_keyword_usage']['score'] < 0.8:
            suggestions.extend(report['primary_keyword_usage']['message'].split(' | '))
            
        # Heading structure optimization
        heading_data = report['heading_distribution']
        if not heading_data['hierarchy_valid']:
            suggestions.append("Fix heading hierarchy - don't skip levels")
        if heading_data['counts']['h1'] != 1:
            suggestions.append("Ensure exactly one H1 heading")
        if sum(heading_data['counts'][f'h{i}'] for i in range(2, 4)) < 3:
            suggestions.append("Add more H2/H3 subheadings for better structure")
            
        # Semantic optimization
        semantic_score = report['semantic_cohesion']
        if semantic_score < self.OPTIMAL_RANGES['semantic_variation'][0]:
            suggestions.append("Increase use of semantic variations and related terms")
            
        # Question coverage
        faq_data = report['faq_presence']
        if faq_data['score'] < 0.7:
            suggestions.append(f"Address more target questions - {faq_data['message']}")
            
        # Calculate overall score
        weights = {
            'primary_keyword_usage': 0.35,
            'semantic_cohesion': 0.25,
            'heading_distribution': 0.2,
            'faq_presence': 0.2
        }
        
        score = (
            report['primary_keyword_usage']['score'] * weights['primary_keyword_usage'] +
            report['semantic_cohesion'] * weights['semantic_cohesion'] +
            (1.0 if report['heading_distribution']['hierarchy_valid'] else 0.5) * weights['heading_distribution'] +
            report['faq_presence']['score'] * weights['faq_presence']
        )

        return {
            'score': round(score, 2),
            'suggestions': suggestions,
            'details': report
        }