
import nltk
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)


import os
import json
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ollama
 
@dataclass
class SimilarityResult:
    """Class to hold detailed similarity analysis results."""
    category: str
    similarity_score: float
    matched_content: str
    content_type: str  # 'word' or 'sentence'
    risk_level: str    # 'high', 'medium', 'low'


@dataclass
class QueryAnalysis:
    """Class to hold comprehensive query analysis."""
    query: str
    is_safe: bool
    flagged_categories: List[str]
    detailed_matches: List[SimilarityResult]
    highest_similarity: float
    primary_category: Optional[str]
    risk_assessment: str
    processing_time: float



class AdvancedQueryMatcher:
    """
    Advanced query matching system with comprehensive categorization,
    confidence scoring, and detailed similarity analysis.
    """
    
    def __init__(self):
        """
            Initialize the matcher with model and data loading.
        """
        self.model = None  # Placeholder for the model, not loaded yet
        self.embeddings = {}  # Dictionary to store embeddings
        self.sentence_embeddings = {}  # Dictionary to store sentence embeddings
        self.category_hierarchies = {}  # Dictionary to store category hierarchy mappings
        
        # Define general risk thresholds
        self.risk_thresholds = {
            'high_risk': 0.75,     # Threshold for high-risk detection
            'medium_risk': 0.65,   # Threshold for medium-risk detection
            'low_risk': 0.55,      # Threshold for low-risk detection
            'safe': 0.45           # Threshold for safe detection
        }
        
        # Define stricter thresholds for critical categories
        self.critical_category_thresholds = {
            'self_harm': 0.65,                      # Threshold for self-harm
            'violence_and_crime': 0.70,            # Threshold for violence/crime
            'weapons_and_warfare': 0.70,           # Threshold for weapons/warfare
            'financial_and_illegal_activities': 0.70,  # Threshold for financial/illegal
            'child_safety': 0.60,                  # Threshold for child safety
            'sexual_content': 0.65                 # Threshold for sexual content
        }
        
        # Whitelist of safe terms that should not be flagged - MORE SAFE TERMS COULD BE ADDED HERE
        self.safe_terms = {
            'geographical': [
                'earth', 'world', 'planet', 'continent', 'country', 'nation', 'location', 'place', 'where', 'map'
            ],
            'educational': [
                'learn', 'study', 'education', 'school', 'university', 'research', 'knowledge', 'information'
            ],
            'general': [
                'help', 'question', 'answer', 'explain', 'describe', 'tell', 'what', 'how', 'why', 'when'
            ]
        }
        self._initialize_system()  # Call system initialization method



    
    def _initialize_system(self):
        """Initialize the entire matching system."""
        self._load_model()  # Load the transformer model
        self._load_embeddings()  # Load pre-computed embeddings
        self._setup_category_hierarchies()  # Setup category hierarchy mappings


        
    
    def _load_model(self):
        """
            Load the sentence transformer model.
        """
        try:
            # Default local path for saved embedding model snapshots
            base_path = "./Embedding_Model/all_mini_embed_model/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/"
            
            # List all directories inside base_path (each represents a commit hash)
            hashes = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
            
            # If no snapshot hash directories are found, raise an error
            if not hashes:
                raise ValueError("No snapshot hash folders found!")
            
            # Pick the latest snapshot folder (last after sorting)
            commit_hash = sorted(hashes)[-1]
            
            # Build the full path to the chosen snapshot
            model_path = os.path.join(base_path, commit_hash)
            
            # Load the SentenceTransformer model from the local snapshot
            self.model = SentenceTransformer(model_path)
        
        except Exception as e:
            # Return exception if any error occurs
            return e


            
    
    def _load_embeddings(self):
        """Load and encode topics and sentences data."""
        try:
            # Path to JSON file containing topics to avoid
            topics_path = "./Files/topics_to_avoid.json"
            
            # Open and load topics JSON file
            with open(topics_path, "r", encoding="utf-8") as f:
                topics_data = json.load(f)
            
            # Path to JSON file containing sentences to avoid
            sentences_path = "./Files/sentences_to_avoid.json"
            
            # Open and load sentences JSON file
            with open(sentences_path, "r", encoding="utf-8") as f:
                sentences_data = json.load(f)
            
            # Encode all words in topics into embeddings and store by category
            for category, words in topics_data.items():
                self.embeddings[category] = self.model.encode(words, convert_to_numpy=True)
            
            # Encode all sentences into embeddings and store by category
            for category, sentences in sentences_data.items():
                self.sentence_embeddings[category] = self.model.encode(sentences, convert_to_numpy=True)
    
        except Exception as e:
            # Return the exception object if an error occurs
            return e


    

            
    def _setup_category_hierarchies(self):
        """
            Setup hierarchical category relationships for better organization.
        """
        # Define dictionary of category hierarchies
        self.category_hierarchies = {
            # Security threats category with child categories and severity level
            'security_threats': {
                'children': ['violence_and_crime', 'weapons_and_warfare', 'terrorism', 'extremism_and_hate_groups'],
                'severity': 'critical'
            },
            # Personal harm category with child categories and severity level
            'personal_harm': {
                'children': ['self_harm', 'toxicity', 'harassment'],
                'severity': 'high'
            },
            # Illegal activities category with child categories and severity level
            'illegal_activities': {
                'children': ['drugs_and_substances', 'fraud_and_scams', 'financial_and_illegal_activities'],
                'severity': 'high'
            },
            # Privacy violations category with child categories and severity level
            'privacy_violations': {
                'children': ['pii', 'cybersecurity_threats'],
                'severity': 'medium'
            },
            # Inappropriate content category with child categories and severity level
            'inappropriate_content': {
                'children': ['sexual_content', 'child_safety'],
                'severity': 'critical'
            },
            # Misinformation category with child categories and severity level
            'misinformation': {
                'children': ['misinformation_and_disinformation'],
                'severity': 'high'
            }
        }

    
    
    
    def _clean_query(self, query: str) -> List[str]:
        """
            Clean and tokenize query, removing stopwords.
        """
        try:
            # Load English stopwords into a set for faster lookup
            stop_words = set(stopwords.words("english"))
            
            # Convert query to lowercase and tokenize into words
            words = word_tokenize(query.lower())
            
            # Filter words: keep only alphabetic tokens, exclude stopwords, and require length > 2
            filtered_words = [w for w in words if w.isalpha() and w not in stop_words and len(w) > 2]
            
            # Return the cleaned list of words
            return filtered_words
            
        except Exception:
            # Fallback: simple split and lowercase if tokenization/stopwords fail
            return [w.lower() for w in query.split() if len(w) > 2]


            
    
    def _is_safe_query(self, query: str) -> bool:
        """
            Check if query appears to be a legitimate educational/geographical question.
        """
        
        # Convert query to lowercase for consistent matching
        query_lower = query.lower()
        
        # List of common educational patterns
        educational_patterns = [
            'where is', 'what is', 'how does', 'explain', 'tell me about',
            'how to learn', 'what does', 'can you explain', 'help me understand'
        ]
        
        # List of geographical keywords
        geographical_indicators = ['country', 'city', 'location', 'capital', 'continent', 'map']
        
        # Common safe query starters
        safe_starters = ['where', 'what', 'how', 'when', 'why', 'can you', 'please explain', 'ask for', 'request']

        # Workplace/professional patterns
        workplace_patterns = [
            'ask for', 'request for', 'apply for', 'need to', 'want to', 'how to apply',
            'leave to', 'time off', 'vacation', 'sick leave', 'hr', 'human resources',
            'days leave', 'days off', 'hours', 'shift', 'work schedule', 'meeting with',
            'talk to', 'speak with', 'contact', 'reach out', 'get in touch',
            'how to ask', 'how do i ask', 'asking for', 'requesting'
        ]

        # OTHER SAFETY STANDARDS CAN BE GIVEN HERE
        
        # Check if query contains any educational patterns
        has_educational_pattern = any(pattern in query_lower for pattern in educational_patterns)
        
        # Check if query contains any geographical indicators
        has_geographical_indicator = any(indicator in query_lower for indicator in geographical_indicators)
        
        # Check if query contains workplace patterns
        has_workplace_pattern = any(pattern in query_lower for pattern in workplace_patterns)
        
        # Check if query starts with a safe phrase
        starts_safely = any(query_lower.startswith(starter) for starter in safe_starters)
        
        # Count overlap of query words with predefined safe terms
        query_words = set(query_lower.split())
        safe_word_count = 0
        for category, terms in self.safe_terms.items():
            safe_word_count += len(query_words.intersection(terms))
        
        # Special check for workplace queries
        workplace_words = query_words.intersection(self.safe_terms.get('workplace', []))
        
        # Query is safe if it matches educational patterns,
        # OR contains geographical indicators,
        # OR contains workplace patterns,
        # OR starts safely and has at least 2 safe words,
        # OR has workplace-related words
        return (has_educational_pattern or 
                has_geographical_indicator or 
                has_workplace_pattern or
                (starts_safely and safe_word_count >= 2) or
                len(workplace_words) >= 2)




    def _is_common_word_match(self, matched_content: str, query: str) -> bool:
        """
            Check if the match is due to a common word that could be innocent.
        """
        
        # Split query into lowercase words and store in a set
        query_words = set(query.lower().split())
        
        # Split matched content into lowercase words and store in a set
        matched_words = set(matched_content.lower().split())
        
        # Initialize an empty set to hold all safe terms
        all_safe_terms = set()
        
        # Collect all safe terms from different categories into one flat set
        for terms in self.safe_terms.values():
            all_safe_terms.update(terms)
        
        # Find words common to both query and matched content
        common_words = query_words.intersection(matched_words)
        
        # Return True if common words exist AND all of them are safe terms
        return len(common_words) > 0 and common_words.issubset(all_safe_terms)


        

        
    def _filter_false_positives(self, results: List[SimilarityResult], query: str) -> List[SimilarityResult]:
        """
            Filter out likely false positives based on query context.
        """
        
        # If the query is recognized as safe
        if self._is_safe_query(query):
            # Initialize an empty list for filtered results
            filtered_results = []
            
            # Loop through each similarity result
            for result in results:
                # If the result belongs to a critical category → lower threshold
                if result.category in self.critical_category_thresholds:
                    min_threshold = 0.75
                else:
                    # For non-critical categories → stricter threshold
                    min_threshold = 0.88
                
                # Keep result only if similarity score passes threshold
                if result.similarity_score >= min_threshold:
                    # For critical categories, keep directly
                    # For non-critical, check it’s not just a common word match
                    if result.category in self.critical_category_thresholds or not self._is_common_word_match(result.matched_content, query):
                        filtered_results.append(result)
            
            # Return the filtered results for safe queries
            return filtered_results
        
        # If query is not safe, return results unfiltered
        return results

    



        
    
    def _calculate_detailed_similarity(self, query: str, threshold: float = 0.45) -> List[SimilarityResult]:
        """
            Calculate detailed similarity scores for both words and sentences.
        """
        
        results = []  # Initialize list to store similarity results
        
        # ---------------- Word-level analysis ----------------
        
        # Clean query into filtered words (remove stopwords, etc.)
        filtered_words = self._clean_query(query)
        
        # If valid filtered words exist
        if filtered_words:
            # Encode query words into embeddings
            query_word_embeddings = self.model.encode(filtered_words, convert_to_numpy=True, show_progress_bar=False)
            
            # Compare query word embeddings with stored risky word embeddings
            for category, word_embeddings in self.embeddings.items():
                for q_emb in query_word_embeddings:
                    # Compute cosine similarity between query word and risky words
                    similarities = cosine_similarity([q_emb], word_embeddings)[0]
                    
                    # Take the maximum similarity score
                    max_sim = np.max(similarities)
                    
                    # If max similarity is above threshold, consider it a match
                    if max_sim >= threshold:
                        # Index of the most similar risky word
                        max_idx = np.argmax(similarities)
                        
                        # Retrieve actual risky word from JSON file
                        try:
                            topics_path = "./Files/topics_to_avoid.json"
                            with open(topics_path, "r") as f:
                                topics_data = json.load(f)
                            matched_word = topics_data[category][max_idx]
                        except:
                            matched_word = "unknown"  # If lookup fails
                        
                        # Append result with details
                        results.append(SimilarityResult(
                            category=category,
                            similarity_score=float(max_sim),
                            matched_content=matched_word,
                            content_type='word',
                            risk_level=self._determine_risk_level(max_sim, category)
                        ))
        
        # ---------------- Sentence-level analysis ----------------
        
        # Encode the entire query as a sentence embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]
        
        # Compare query sentence embedding with stored risky sentence embeddings
        for category, sentence_embeddings in self.sentence_embeddings.items():
            # Compute cosine similarities
            similarities = cosine_similarity([query_embedding], sentence_embeddings)[0]
            
            # Take maximum similarity score
            max_sim = np.max(similarities)
            
            # If max similarity exceeds threshold
            if max_sim >= threshold:
                # Index of the most similar risky sentence
                max_idx = np.argmax(similarities)
                
                # Retrieve actual risky sentence from JSON file
                try:
                    sentences_path = "./Files/sentences_to_avoid.json"
                    with open(sentences_path, "r") as f:
                        sentences_data = json.load(f)
                    matched_sentence = sentences_data[category][max_idx]
                except:
                    matched_sentence = "unknown"  # If lookup fails
                
                # Append result with details
                results.append(SimilarityResult(
                    category=category,
                    similarity_score=float(max_sim),
                    matched_content=matched_sentence,
                    content_type='sentence',
                    risk_level=self._determine_risk_level(max_sim, category)
                ))
        
        # ---------------- Post-processing ----------------
        
        # Sort results by similarity score in descending order
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Apply false positive filtering for safer results
        filtered_results = self._filter_false_positives(results, query)
        
        # Return the final filtered list of similarity results
        return filtered_results
        



        
    def _determine_risk_level(self, similarity_score: float, category: str = None) -> str:
        """Determine risk level based on similarity score and category-specific thresholds."""
        
        # If category is critical, use its specific threshold
        if category and category in self.critical_category_thresholds:
            critical_threshold = self.critical_category_thresholds[category]
            if similarity_score >= critical_threshold:
                return 'high'  # Elevate critical category matches to high risk
        
        # Standard threshold logic for non-critical categories
        if similarity_score >= self.risk_thresholds['high_risk']:
            return 'high'  # High similarity → high risk
        elif similarity_score >= self.risk_thresholds['medium_risk']:
            return 'medium'  # Medium similarity → medium risk
        elif similarity_score >= self.risk_thresholds['low_risk']:
            return 'low'  # Low similarity → low risk
        else:
            return 'minimal'  # Below all thresholds → minimal risk




    
    def _get_category_severity(self, category: str) -> str:
        """Get severity level for a category based on hierarchies."""
        for parent, info in self.category_hierarchies.items():
            if category in info['children']:
                return info['severity']
        return 'medium'  # default



        
    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Perform comprehensive analysis of a user query.
        Returns detailed analysis including categories, similarities, and risk assessment.
        """
        
        # Record start time for processing time calculation
        start_time = time.time()
        
        # Compute detailed similarity scores for the query using a balanced threshold
        similarity_results = self._calculate_detailed_similarity(query, threshold=0.60)
        
        # Extract unique categories from similarity results that were flagged
        flagged_categories = list(set([result.category for result in similarity_results]))
        
        # Identify high and medium risk matches
        high_risk_matches = [r for r in similarity_results if r.risk_level in ['high', 'medium']]
        
        # Query is considered safe if no high/medium risk matches exist
        is_safe = len(high_risk_matches) == 0
        
        # Get the highest similarity score among all matches
        highest_similarity = similarity_results[0].similarity_score if similarity_results else 0.0
        
        # Determine the primary category with the highest similarity
        primary_category = similarity_results[0].category if similarity_results else None
        
        # Generate a summarized risk assessment based on flagged categories and similarity results
        risk_assessment = self._generate_risk_assessment(similarity_results, flagged_categories)
        
        # Calculate total processing time
        processing_time = time.time() - start_time
        
        # Return a structured QueryAnalysis object with all relevant information
        return QueryAnalysis(
            query=query,
            is_safe=is_safe,
            flagged_categories=flagged_categories,
            detailed_matches=similarity_results[:10],  # Only include top 10 matches
            highest_similarity=highest_similarity,
            primary_category=primary_category,
            risk_assessment=risk_assessment,
            processing_time=processing_time
        )





    
    def _generate_risk_assessment(self, similarity_results: List[SimilarityResult], flagged_categories: List[str]) -> str:
        """Generate comprehensive risk assessment text."""
        
        # If no similarity matches found, return safe
        if not similarity_results:
            return "Safe - No concerning content detected"
        
        # Separate high-risk matches
        high_risk = [r for r in similarity_results if r.risk_level == 'high']
        
        # Separate medium-risk matches
        medium_risk = [r for r in similarity_results if r.risk_level == 'medium']
        
        # If high-risk match exists, report it
        if high_risk:
            primary_cat = high_risk[0].category.replace('_', ' ').title()  # Format category for readability
            return f"High Risk - Detected {primary_cat} content (confidence: {high_risk[0].similarity_score:.2f})"
        
        # If medium-risk match exists, report it
        elif medium_risk:
            primary_cat = medium_risk[0].category.replace('_', ' ').title()  # Format category
            return f"Medium Risk - Potential {primary_cat} content (confidence: {medium_risk[0].similarity_score:.2f})"
        
        # Otherwise, report as low risk with minor flagged categories
        else:
            return f"Low Risk - Minor concerns detected in {len(flagged_categories)} categories"






            
    def get_category_breakdown(self, query: str) -> Dict[str, Dict[str, Any]]:
        """Get detailed breakdown by category with statistics."""
        
        # Perform comprehensive analysis of the query
        analysis = self.analyze_query(query)
        
        # Initialize category statistics dictionary with default structure
        category_stats = defaultdict(lambda: {
            'matches': [],            # List of similarity results
            'max_similarity': 0.0,    # Maximum similarity score in this category
            'avg_similarity': 0.0,    # Average similarity score
            'risk_level': 'minimal',  # Highest risk level in this category
            'severity': 'low'         # Severity level of category
        })
        
        # Populate statistics per category based on detailed matches
        for result in analysis.detailed_matches:
            cat = result.category
            
            # Append match to category list
            category_stats[cat]['matches'].append(result)
            
            # Update maximum similarity score
            category_stats[cat]['max_similarity'] = max(
                category_stats[cat]['max_similarity'], 
                result.similarity_score
            )
            
            # Retrieve and assign category severity
            category_stats[cat]['severity'] = self._get_category_severity(cat)
            
            # Update category risk level to the highest among matches
            current_risk = category_stats[cat]['risk_level']
            new_risk = result.risk_level
            risk_order = ['minimal', 'low', 'medium', 'high']
            if risk_order.index(new_risk) > risk_order.index(current_risk):
                category_stats[cat]['risk_level'] = new_risk
        
        # Calculate average similarity for each category
        for cat, stats in category_stats.items():
            if stats['matches']:
                stats['avg_similarity'] = np.mean([m.similarity_score for m in stats['matches']])
        
        # Convert defaultdict to regular dict and return
        return dict(category_stats)

    
 
def print_detailed_results(results: List[Dict[str, Any]]):
    """Print detailed analysis results in a formatted way."""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE QUERY ANALYSIS RESULTS")
    print("="*80)
    
    for i, result in enumerate(results, 1):
        print(f"\n--- Query {i} ---")
        print(f"Query: {result['query']}")
        print(f"Status: {result.get('status', 'unknown').upper()}")
        
        if 'analysis' in result:
            analysis = result['analysis']
            print(f"Safety: {'SAFE' if analysis.is_safe else 'FLAGGED'}")
            print(f"Risk Assessment: {analysis.risk_assessment}")
            
            if analysis.flagged_categories:
                print(f"Flagged Categories: {', '.join(analysis.flagged_categories)}")
            
            if analysis.detailed_matches:
                print("Top Matches:")
                for match in analysis.detailed_matches[:3]:
                    print(f"  • {match.category}: {match.similarity_score:.3f} ({match.risk_level} risk)")
        
        print(f"Response: {result['response'][:200]}{'...' if len(result['response']) > 200 else ''}")
        print("-" * 60)

