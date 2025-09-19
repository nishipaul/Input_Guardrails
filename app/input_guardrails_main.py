import warnings
warnings.filterwarnings("ignore")

from typing import Dict, Any, List
from app.pii_detector import AdvancedPIIValidator
from app.safety_check import AdvancedQueryMatcher
import time
import ollama


from guardrails import Guard
matcher = AdvancedQueryMatcher()
guard = Guard()
guard.use_many(AdvancedPIIValidator())



def process_query_comprehensive(query: str) -> Dict[str, Any]:
    """
    Main function to process a query with comprehensive analysis and response.
    """
    
    # Perform full query analysis
    analysis = matcher.analyze_query(query)
    
    # Get detailed breakdown by category
    category_breakdown = matcher.get_category_breakdown(query)
    
    # Initialize response dictionary
    response = {
        'query': query,
        'analysis': analysis,
        'category_breakdown': category_breakdown,
        'response': ''
    }

    # Case 1: Query is safe and similarity is below low-risk threshold
    if analysis.is_safe and analysis.highest_similarity < matcher.risk_thresholds['low_risk']:
        response['response'] = 'SAFE'
        response['status'] = 'approved'
    
    # Case 2: Query has high-risk similarity
    else:        
        if analysis.highest_similarity >= matcher.risk_thresholds['high_risk']:
            response['response'] = (
                f"I cannot assist with this request as it relates to {analysis.primary_category.replace('_', ' ')} "
                f"content. Please ensure your queries are appropriate and safe."
            )
            response['status'] = 'blocked'
        
        # Case 3: Medium-risk query - cautious guidance
        else:
            response['response'] = (
                f"I notice your query may relate to {analysis.primary_category.replace('_', ' ')} topics. "
                f"I can provide general, educational information while ensuring safety and appropriateness."
            )
            response['status'] = 'cautious'
            response['status'] = 'cautious'
    
    # Return final structured response
    return response




def process_query(query: str) -> Dict[str, Any]:
    # Define function signature: accepts a query string and returns a result dictionary
    """Process a single query with matcher and Guard validation."""
    # Docstring: short description of the function purpose

    try:
        # Start a try block to catch unexpected runtime errors during processing
        matcher_result = process_query_comprehensive(query)
        # Call the comprehensive matcher to analyze the query; expect a dict with keys like 'status', 'analysis', 'response'
        

        # --- BLOCKED branch ---
        if matcher_result.get("status", "").upper() == "BLOCKED":
            # Normalize status string and check if result indicates a blocked query
            analysis = matcher_result.get("analysis")
            # Extract the analysis object from the matcher result (likely a QueryAnalysis object)

            return {
                # Return a dictionary immediately for blocked queries
                "query": query,  # Echo the original query
                "status": "blocked",  # Human-friendly status
                "analysis": {
                    # Provide a compact analysis summary for callers
                    "is_safe": analysis.is_safe,
                    "risk_assessment": analysis.risk_assessment,
                    "flagged_categories": analysis.flagged_categories
                },
                "response": matcher_result.get("response", "")
                # Include the text response from the matcher if present (empty string fallback)
            }

        # --- CAUTIOUS branch ---
        elif matcher_result.get("status", "").upper() == "CAUTIOUS":
            # Check if the matcher classified the query as cautious
            # PEVENTING GUARDRAIL CHECK IF DETECTED HERE
            analysis = matcher_result.get("analysis")
            # Extract analysis object for inclusion in the returned summar

            return {
                # Return a cautious response summary to caller
                "query": query,  # Echo original query
                "status": "cautious",  # Human-friendly status
                "analysis": {
                    # Provide compact analysis summary
                    "is_safe": analysis.is_safe,
                    "risk_assessment": analysis.risk_assessment,
                    "flagged_categories": analysis.flagged_categories
                },
                "response": matcher_result.get("response", "")
                # Include matcher's response text (or empty string)
            }

        # --- APPROVED branch ---
        elif matcher_result.get("status", "").upper() == "APPROVED":
            # If matcher approves the request, perform guardrail validation
            try:
                guard.validate(query)
                # Call guard.validate to perform policy/guardrail checks on the query; may raise an exception on failure
                matcher_result["guard_validation"] = "passed"
                # If no exception raised, mark guard validation as passed in the matcher_result dict
            except Exception as e:
                # If guard validation raised an exception, capture it here
                matcher_result["guard_validation"] = f"failed - {str(e)}"
                # Annotate matcher_result with the failure reason
                matcher_result = check_guardrail_result(matcher_result)

                # Optionally mutate/augment matcher_result via check_guardrail_result to reflect guard findings

            return matcher_result
            # Return the (possibly updated) matcher_result for approved queries after guard validation

    except Exception as e:
        # Top-level exception handler: catch any unexpected error that occurred above
        return {
            # Return a standardized error response
            "query": query,  # Echo original query
            "status": "error",  # Status indicating an internal error
            "error": str(e)  # Stringified exception message for debugging
        }




def check_guardrail_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Check the processed query result for guardrail failures."""
    
    
    # Extract the current status of the result (default "unknown" if missing).
    status = result.get("status", "unknown").lower()
    
    # Extract the guard validation result (default empty string).
    # Convert to lowercase for consistent comparison.
    guard_validation = result.get("guard_validation", "").lower()
    

    # --- Case 1: Guardrail validation failed ---
    if "failed" in guard_validation:
        # If guard validation indicates failure, override the result.
        return {
            "query": result.get("query"),  # Original query text
            "status": "rejected",  # Explicit rejection status
            "reason": guard_validation,  # Store failure reason for debugging
            "message": "Query rejected due to guardrail validation failure."
            # User-facing message explaining rejection
        }

    # --- Case 2: Query was BLOCKED earlier ---
    if status == "blocked":
        # Preserve the original blocked status, analysis, and response
        return {
            "query": result.get("query"),     # Original query
            "status": "blocked",              # Keep blocked status
            "analysis": result.get("analysis"),  # Risk/analysis info from matcher
            "response": result.get("response")   # Response provided by matcher
        }

    # --- Case 3: Otherwise, treat as APPROVED ---
    return {
        "query": result.get("query"),  # Original query
        "status": "approved",  # Mark as approved since no failures/blocking found
        "response": result.get("response", ""),  # Safe response (fallback empty string)
        "guard_validation": guard_validation or "passed"
        # If guard_validation not explicitly set, assume "passed"
    }



def call_model_safely(query: str) -> str:
    """Call the language model with safety checks."""
    try:
        # Build conversation history for the model
        messages = [
            {
                "role": "system",   # System prompt sets rules for model behavior
                "content": (
                    "You are a helpful and safe assistant. "
                    "Answer only valid, safe, and appropriate questions. "
                    "If the query involves harmful, illegal, or inappropriate content, "
                    "politely decline and explain why you cannot help. "
                    "Be clear, concise, and professional."
                )
            },
            {
                "role": "user",    # Actual user query
                "content": query
            }
        ]
        
        # Call Ollama model with structured messages
        response = ollama.chat(
            model="llama3:8b",     # Chosen model
            messages=messages,     # Messages (system + user)
            options={"temperature": 0.0}  # Deterministic output for safety
        )
        
        # Extract model's generated text from response object
        return response['message']['content']
    
    except Exception as e:
        # Return a polite fallback message to the user
        return "I apologize, but I'm unable to process your request at this time."





def execute_if_safe(result: Dict[str, Any]) -> Dict[str, Any]:
    """Execute call_model_safely only if the response is safe and guard validation passed.
    Otherwise return reason for not completing."""
    
    # Extract key fields from the result dict
    analysis = result.get("analysis")                        # Analysis object/dict with risk info
    status = result.get("status", "").lower()               # Status string: approved / blocked / cautious / error

    if 'reason' in result:
        guard_validation = result['reason']
    else:
        guard_validation = result.get("guard_validation", "passed").lower()  # Guard check result: passed / failed
    

    # Extract risk assessment text if analysis is present
    if isinstance(analysis, dict):
        risk_assessment = analysis.get("risk_assessment", "not available").lower()
    elif hasattr(analysis, "risk_assessment"):
        risk_assessment = getattr(analysis, "risk_assessment", "not available").lower()
    else:
        risk_assessment = "not available"

    
    # Allow execution for approved queries or safe cautious queries (like workplace questions)
    if (
        analysis                               # Analysis must exist
        and (
            (getattr(analysis, "is_safe", False) and status == "approved") or  # Safe approved queries
            (status == "cautious" and guard_validation == "passed" and 
             getattr(analysis, "risk_assessment", "").lower().startswith("low risk"))  # Low-risk cautious queries
        )
    ):
        # Safe to execute → call the model
        return {
            "query": result.get("query"),
            "status": "completed",
            "response": call_model_safely(result["query"])  # Call model safely
        }
    else:
        # Block execution → return structured reason
        return {
            "query": result.get("query"),
            "status": "not_completed",   # Explicitly mark as blocked
            "reason": {
                "status": status,                   # Whether blocked/cautious/error
                "guard_validation": guard_validation,  # Guard failure reason
                "risk_assessment": risk_assessment     # Why it was flagged
            },
            "message": "Query not safe to process, execution blocked."
        }
def process_query_batch(queries: List[str]) -> List[Dict[str, Any]]:
    """Process multiple queries sequentially with matcher and Guard validation."""
    start_time = time.time()
    results = [process_query(query) for query in queries]
    model_call = [execute_if_safe(response) for response in results]
    end_time = time.time()
    #logger.info(f"Processed {len(queries)} queries in {end_time - start_time:.2f} seconds")
    return model_call
