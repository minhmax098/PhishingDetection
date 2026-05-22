import json
import re

def sanitize_input(url):
    """
    Sanitizes the input URL to prevent prompt injection or execution of malicious payloads in the LLM.
    """
    # Remove HTML tags or script injections
    sanitized = re.sub(r'<[^>]*>', '', str(url))
    # Replace quotes and backticks to prevent breaking out of JSON
    sanitized = sanitized.replace('"', '&quot;').replace("'", '&apos;').replace('`', '&#96;')
    return sanitized

def create_structured_prompt(sanitized_url, prediction, confidence, lime_evidence, shap_evidence, domain_features):
    """
    Creates a JSON-based structured prompt for the LLM.
    """
    prompt_data = {
        "task": "Explain phishing detection prediction and provide security reasoning.",
        "input": {
            "url": sanitized_url,
            "model_prediction": "phishing" if prediction == 1 else "legitimate",
            "confidence_score": float(confidence),
            "evidence": {
                "lime_top_features": lime_evidence,
                "shap_top_features": shap_evidence,
                "domain_info": domain_features
            }
        },
        "instructions": [
            "Analyze the provided evidence.",
            "Generate an analyst-style explanation.",
            "Identify suspicious pattern categories.",
            "Provide a risk rationale.",
            "Recommend an action.",
            "Flag any uncertainty in the model's prediction."
        ],
        "output_format_expected": {
            "analyst_style_explanation": "string",
            "suspicious_pattern_category": "string",
            "risk_rationale": "string",
            "recommended_action": "string",
            "uncertainty_flag": "boolean"
        }
    }
    return json.dumps(prompt_data, indent=2)

def mock_llm_interface(prompt_json):
    """
    A mock interface to simulate an LLM response (to be replaced by actual OpenAI API calls).
    """
    try:
        data = json.loads(prompt_json)
        prediction = data["input"]["model_prediction"]
        url = data["input"]["url"]
    except json.JSONDecodeError:
        return '{"error": "Invalid prompt format."}'

    if prediction == "phishing":
        response = {
            "analyst_style_explanation": f"The URL '{url}' exhibits multiple suspicious indicators including structural obfuscation and high domain entropy, commonly associated with malicious activities.",
            "suspicious_pattern_category": "Domain Obfuscation / Brand Impersonation",
            "risk_rationale": "High risk due to suspicious lexical features and lack of reliable domain reputation.",
            "recommended_action": "Block access to the URL and notify the security team.",
            "uncertainty_flag": False
        }
    else:
        response = {
            "analyst_style_explanation": f"The URL '{url}' appears to follow standard structural patterns with no significant malicious indicators detected by the model.",
            "suspicious_pattern_category": "None",
            "risk_rationale": "Low risk as domain and lexical features align with benign traffic.",
            "recommended_action": "Allow access.",
            "uncertainty_flag": False
        }

    return json.dumps(response, indent=2)

def validate_llm_output(llm_response_json, original_prediction):
    """
    Validates that the LLM's explanation is consistent with the model's original prediction.
    """
    try:
        response = json.loads(llm_response_json)
        # Simple rule-based validation
        explanation = response.get("analyst_style_explanation", "").lower()

        if original_prediction == 1: # Phishing
            if "benign" in explanation or "legitimate" in explanation and "not" not in explanation:
                return False, "Validation Failed: Explanation contradicts phishing prediction."
        else: # Legitimate
            if "malicious" in explanation or "phishing" in explanation and "not" not in explanation:
                return False, "Validation Failed: Explanation contradicts legitimate prediction."

        return True, response
    except json.JSONDecodeError:
        return False, "Validation Failed: Invalid JSON output from LLM."

def run_llm_pipeline(url, prediction, confidence, lime_evidence, shap_evidence, domain_features):
    """
    Executes the full Secure LLM Reasoning Pipeline.
    """
    sanitized_url = sanitize_input(url)
    prompt = create_structured_prompt(sanitized_url, prediction, confidence, lime_evidence, shap_evidence, domain_features)
    llm_output = mock_llm_interface(prompt)
    is_valid, final_result = validate_llm_output(llm_output, prediction)

    if is_valid:
        return final_result
    else:
        # Fallback explanation
        return {
            "analyst_style_explanation": "Automated explanation failed validation. Model predicted: " + ("Phishing" if prediction == 1 else "Legitimate"),
            "error_details": final_result
        }

if __name__ == "__main__":
    print("LLM Reasoning Pipeline module ready.")
    res = run_llm_pipeline("http://<script>alert(1)</script>.phish.com", 1, 0.95, {}, {}, {"domain_age_days": 2})
    print("Test Output:")
    print(json.dumps(res, indent=2))
