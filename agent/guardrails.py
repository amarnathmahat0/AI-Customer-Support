"""
agent/guardrails.py — Output validation for LLM responses.

Validates agent responses against:
  1. No hallucinated order numbers (must be ORD-XXXXXX format)
  2. Professional tone (no aggressive/dismissive language)
  3. Length constraint (< MAX_RESPONSE_WORDS words)

Implements retry logic with a configurable maximum retry count.
"""

import re
import logging
import os
from typing import Tuple, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Validation patterns
ORDER_ID_PATTERN = re.compile(r'\bORD-\d{6}\b')
# Exclude the literal placeholder 'ORD-XXXXXX' used in prompt templates
INVALID_ORDER_PATTERN = re.compile(r'\bORD-(?!\d{6}\b)(?!XXXXXX\b)\S+')

# Aggressive/dismissive language patterns
PROBLEMATIC_PHRASES = [
    r'\bstupid\b', r'\bidiot\b', r'\bmoron\b', r'\bdumb\b',
    r'\byour fault\b', r'\bbother us\b', r"\bdon't care\b",
    r'\bnot my problem\b', r'\bnot our problem\b', r'\bjust deal\b',
    r'\bfigure it out\b', r'\bi don\'t know\b', r'\bcannot help\b',
    r'\bimpossible\b', r'\bnever\b', r'\babsolutely not\b',
    r'\btoo bad\b', r'\bwhatever\b', r'\bwho cares\b',
]

PROBLEMATIC_RE = re.compile(
    '|'.join(PROBLEMATIC_PHRASES),
    re.IGNORECASE
)


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    passed: bool
    check_name: str
    message: str
    severity: str = "error"  # "error" or "warning"


@dataclass
class GuardrailResult:
    """Complete result of running all guardrail validations."""
    passed: bool
    retry_needed: bool
    validations: List[ValidationResult]
    retry_instruction: Optional[str]
    word_count: int


def get_max_words() -> int:
    """Return the maximum allowed response word count from env."""
    return int(os.getenv("MAX_RESPONSE_WORDS", "200"))


def get_max_retries() -> int:
    """Return the maximum number of guardrail retries from env."""
    return int(os.getenv("GUARDRAILS_MAX_RETRIES", "2"))


def validate_order_numbers(text: str) -> ValidationResult:
    """
    Validate that all order numbers in the response follow ORD-XXXXXX format.
    
    Checks for:
    - Malformed order IDs (e.g., ORD-123, ORD-ABCDEF)
    - Valid format: ORD- followed by exactly 6 digits
    
    Args:
        text: The LLM response text to validate
        
    Returns:
        ValidationResult indicating pass/fail and details.
    """
    # Find any malformed order references
    invalid_matches = INVALID_ORDER_PATTERN.findall(text)
    
    if invalid_matches:
        return ValidationResult(
            passed=False,
            check_name="order_number_format",
            message=f"Invalid order ID format found: {invalid_matches}. Use ORD-XXXXXX (6 digits).",
            severity="error"
        )
    
    return ValidationResult(
        passed=True,
        check_name="order_number_format",
        message="Order number format check passed"
    )


def validate_professional_tone(text: str) -> ValidationResult:
    """
    Validate that the response maintains professional, empathetic tone.
    
    Scans for aggressive, dismissive, or unprofessional language patterns
    that would be inappropriate in a customer support context.
    
    Args:
        text: The LLM response text to validate
        
    Returns:
        ValidationResult indicating pass/fail and details.
    """
    matches = PROBLEMATIC_RE.findall(text)
    
    if matches:
        return ValidationResult(
            passed=False,
            check_name="professional_tone",
            message=f"Unprofessional language detected: {matches}. Use empathetic, helpful tone.",
            severity="error"
        )
    
    return ValidationResult(
        passed=True,
        check_name="professional_tone",
        message="Tone check passed"
    )


def validate_response_length(text: str) -> ValidationResult:
    """
    Validate that the response is within the allowed word count.
    
    Args:
        text: The LLM response text to validate
        
    Returns:
        ValidationResult indicating pass/fail with word count details.
    """
    max_words = get_max_words()
    word_count = len(text.split())
    
    if word_count > max_words:
        return ValidationResult(
            passed=False,
            check_name="response_length",
            message=f"Response is {word_count} words, exceeds maximum of {max_words}. Be more concise.",
            severity="warning"
        )
    
    return ValidationResult(
        passed=True,
        check_name="response_length",
        message=f"Length check passed ({word_count}/{max_words} words)"
    )


def validate_no_empty_response(text: str) -> ValidationResult:
    """
    Validate that the response is not empty or trivially short.
    
    Args:
        text: The LLM response text to validate
        
    Returns:
        ValidationResult indicating pass/fail.
    """
    if not text or not text.strip() or len(text.strip()) < 5:
        return ValidationResult(
            passed=False,
            check_name="non_empty_response",
            message="Response is empty or too short to be helpful.",
            severity="error"
        )
    
    return ValidationResult(
        passed=True,
        check_name="non_empty_response",
        message="Non-empty check passed"
    )


def run_guardrails(text: str) -> GuardrailResult:
    """
    Run all validation checks on an LLM response.
    
    Executes all validators and aggregates results into a single
    GuardrailResult. Builds a retry instruction if any checks fail.
    
    Args:
        text: The LLM response text to validate
        
    Returns:
        GuardrailResult with overall pass/fail status, per-check results,
        and a retry instruction if regeneration is needed.
    """
    word_count = len(text.split()) if text else 0
    
    # Run all validators
    checks = [
        validate_no_empty_response(text),
        validate_order_numbers(text),
        validate_professional_tone(text),
        validate_response_length(text),
    ]
    
    failed = [c for c in checks if not c.passed]
    passed = len(failed) == 0
    
    # Build retry instruction from failures
    retry_instruction = None
    if not passed:
        issues = "; ".join(c.message for c in failed)
        retry_instruction = (
            f"Previous response failed validation. Issues: {issues}. "
            f"Please regenerate the response addressing these issues. "
            f"Keep response under {get_max_words()} words, use professional tone, "
            f"and only reference real order IDs in ORD-XXXXXX format."
        )
        logger.warning(f"Guardrail validation failed: {issues}")
    else:
        logger.debug(f"Guardrail validation passed ({word_count} words)")
    
    return GuardrailResult(
        passed=passed,
        retry_needed=not passed,
        validations=checks,
        retry_instruction=retry_instruction,
        word_count=word_count
    )


def validate_with_retry(
    generate_fn,
    base_prompt: str,
    max_retries: int = None
) -> Tuple[str, int]:
    """
    Run guardrail validation with automatic retry on failure.
    
    Calls generate_fn to produce a response, validates it, and if it
    fails, injects the retry instruction into the prompt and tries again.
    
    Args:
        generate_fn: Callable that takes a prompt string and returns response text
        base_prompt: The original prompt to generate a response for
        max_retries: Maximum retry attempts (defaults to GUARDRAILS_MAX_RETRIES env)
        
    Returns:
        Tuple of (response_text, retry_count)
    """
    if max_retries is None:
        max_retries = get_max_retries()
    
    retry_count = 0
    current_prompt = base_prompt
    
    for attempt in range(max_retries + 1):
        try:
            response = generate_fn(current_prompt)
        except Exception as e:
            logger.error(f"Generation failed on attempt {attempt + 1}: {e}")
            if attempt == max_retries:
                return (
                    "I apologize, but I'm experiencing technical difficulties. "
                    "Please try again in a moment or contact our support team directly.",
                    retry_count
                )
            continue
        
        result = run_guardrails(response)
        
        if result.passed:
            logger.info(f"Guardrails passed on attempt {attempt + 1} ({retry_count} retries)")
            return response, retry_count
        
        if attempt < max_retries:
            retry_count += 1
            logger.info(f"Guardrail retry {retry_count}/{max_retries}: {result.retry_instruction[:100]}")
            current_prompt = f"{base_prompt}\n\n[CORRECTION NEEDED]: {result.retry_instruction}"
        else:
            # Max retries exhausted — return the last response anyway with a note
            logger.warning(f"Guardrails failed after {max_retries} retries — returning best effort response")
            return response, retry_count
    
    return (
        "I'm sorry, I encountered an issue generating a response. "
        "Please try rephrasing your question.",
        retry_count
    )
