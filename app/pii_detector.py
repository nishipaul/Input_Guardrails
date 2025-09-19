import re
from typing import List, Tuple, Optional
from guardrails import Validator
from guardrails.validators import register_validator, FailResult, PassResult


@register_validator(name="AdvancedPIIValidator", data_type="string")
class AdvancedPIIValidator(Validator):
    """
    Comprehensive PII validator with enhanced security coverage for multiple types of sensitive information.
    Returns 'Safe' when no PII is detected, otherwise returns specific risk information.
    """

    # Safe words list moved outside the function
    SAFE_ADDRESS_CONTEXT = [
        "leave",
        "holiday",
        "vacation",
        "hr",
        "manager",
        "work"
    ]
    
    def __init__(self):
        """Initialize all regex patterns for different types of PII detection."""
        super().__init__()
        self._init_patterns()
    
    def _init_patterns(self):
        """Initialize comprehensive regex patterns for various PII types."""
        # Email patterns (enhanced)
        self.email_patterns = [
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        ]
        
        # Phone number patterns (global coverage)
        self.phone_patterns = [
            r"\+?\d[\d\s\-\(\)]{7,}\d",  # International format
            r"\(\d{3}\)\s*\d{3}[\-\s]?\d{4}",  # US format (123) 456-7890
            r"\d{3}[\-\s]?\d{3}[\-\s]?\d{4}",  # US format 123-456-7890
            r"\+\d{1,3}\s?\d{1,4}\s?\d{1,4}\s?\d{1,9}",  # International
        ]
        
        # Credit card patterns (all major types)
        self.credit_card_patterns = [
            r"\b4\d{3}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b",  # Visa
            r"\b5[1-5]\d{2}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b",  # Mastercard
            r"\b3[47]\d{2}[\s\-]?\d{6}[\s\-]?\d{5}\b",  # American Express
            r"\b6011[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b",  # Discover
            r"\b(?:\d{4}[\s\-]?){3}\d{4}\b"  # Generic 16-digit
        ]
        
        # Social Security Numbers
        self.ssn_patterns = [
            r"\b\d{3}[\-\s]?\d{2}[\-\s]?\d{4}\b",  # XXX-XX-XXXX or XXX XX XXXX
            r"\b\d{9}\b"  # XXXXXXXXX
        ]
        
        # Government ID patterns
        self.government_id_patterns = [
            r"\b[A-Z]{2}\d{6,8}\b",  # Passport (2 letters + 6-8 digits)
            r"\b[A-Z]\d{7,8}\b",  # Driver's License format
            r"\b\d{4}\s?\d{4}\s?\d{4}\b",  # Aadhar (India)
            r"\b[A-Z]{5}\d{4}[A-Z]\b",  # PAN (India)
            r"\b[A-Z]{2}\d{2}\s?[A-Z]{4}\s?\d{2}\s?[A-Z]\d{3}[A-Z]\b"  # IBAN format
        ]
        
        # Bank account patterns
        self.bank_account_patterns = [
            r"\b\d{8,17}\b",  # Account numbers
            r"\b\d{9}\b",  # Routing numbers (US)
            r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b"  # IBAN
        ]
        
        # Address patterns
        self.address_patterns = [
            r"\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Circle|Cir|Court|Ct)",
            r"\b\d{5}(?:-\d{4})?\b",  # ZIP codes
            r"\b[A-Z]\d[A-Z]\s?\d[A-Z]\d\b"  # Canadian postal codes
        ]
        
        # Date of Birth patterns
        self.dob_patterns = [
            r"\b(0[1-9]|1[0-2])/(0[1-9]|[12]\d|3[01])/\d{4}\b",  # MM/DD/YYYY
            r"\b(0[1-9]|[12]\d|3[01])/(0[1-9]|1[0-2])/\d{4}\b",  # DD/MM/YYYY
            r"\b\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b",  # YYYY-MM-DD
            r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}\b"  # Month DD, YYYY
        ]
        
        # IP address patterns
        self.ip_patterns = [
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b",  # IPv4
            r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"  # IPv6 (simplified)
        ]
        
        # Medical identifiers
        self.medical_patterns = [
            r"\b\d{3}-\d{2}-\d{4}\b",  # Medical record numbers
            r"\bMRN\s*:?\s*\d+\b",  # Medical Record Number
            r"\bDOB\s*:?\s*\d{1,2}/\d{1,2}/\d{4}\b"  # DOB labels
        ]
        
        # Financial patterns
        self.financial_patterns = [
            r"\b\d{1,3}(,\d{3})*\.\d{2}\b",  # Currency amounts
            r"\$\d+(\.\d{2})?",  # Dollar amounts
            r"\bAccount\s*:?\s*\d+\b"  # Account references
        ]

    def _check_emails(self, text: str) -> Optional[str]:
        for pattern in self.email_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return "Email address detected"
        return None

    def _check_phone_numbers(self, text: str) -> Optional[str]:
        for pattern in self.phone_patterns:
            if re.search(pattern, text):
                return "Phone number detected"
        return None

    def _check_credit_cards(self, text: str) -> Optional[str]:
        for pattern in self.credit_card_patterns:
            if re.search(pattern, text):
                return "Credit card number detected"
        return None

    def _check_ssn(self, text: str) -> Optional[str]:
        for pattern in self.ssn_patterns:
            if re.search(pattern, text):
                return "Social Security Number detected"
        return None

    def _check_government_ids(self, text: str) -> Optional[str]:
        for pattern in self.government_id_patterns:
            if re.search(pattern, text):
                return "Government ID number detected"
        return None

    def _check_bank_accounts(self, text: str) -> Optional[str]:
        for pattern in self.bank_account_patterns:
            if re.search(pattern, text):
                return "Bank account information detected"
        return None

    def _check_addresses(self, text: str) -> Optional[str]:
        """Check for physical addresses (skip for HR/leave/work-related queries)."""
        if any(re.search(rf"\b{word}\b", text, re.IGNORECASE) for word in self.SAFE_ADDRESS_CONTEXT):
            return None
        for pattern in self.address_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return "Physical address detected"
        return None

    def _check_dates_of_birth(self, text: str) -> Optional[str]:
        for pattern in self.dob_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return "Date of birth detected"
        return None

    def _check_ip_addresses(self, text: str) -> Optional[str]:
        for pattern in self.ip_patterns:
            if re.search(pattern, text):
                return "IP address detected"
        return None

    def _check_medical_info(self, text: str) -> Optional[str]:
        for pattern in self.medical_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return "Medical information detected"
        return None

    def _check_financial_info(self, text: str) -> Optional[str]:
        for pattern in self.financial_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return "Financial information detected"
        return None

    def _check_sensitive_keywords(self, text: str) -> Optional[str]:
        sensitive_keywords = [
            r"\bpassword\s*:?\s*\w+",
            r"\busername\s*:?\s*\w+",
            r"\bpin\s*:?\s*\d+",
            r"\bsecurity\s+question",
            r"\bmother'?s\s+maiden\s+name",
            r"\bfirst\s+pet\s+name",
            r"\bbirthplace"
        ]
        
        for pattern in sensitive_keywords:
            if re.search(pattern, text, re.IGNORECASE):
                return "Sensitive authentication information detected"
        return None

    def validate(self, text: str, metadata=None):
        if not text or not isinstance(text, str):
            return PassResult()

        check_functions = [
            self._check_emails,
            self._check_phone_numbers, 
            self._check_credit_cards,
            self._check_ssn,
            self._check_government_ids,
            self._check_bank_accounts,
            self._check_addresses,
            self._check_dates_of_birth,
            self._check_ip_addresses,
            self._check_medical_info,
            self._check_financial_info,
            self._check_sensitive_keywords
        ]

        detected_issues = []
        for check_func in check_functions:
            result = check_func(text)
            if result:
                detected_issues.append(result)

        if detected_issues:
            return FailResult(
                error_message=f"PII Risk Detected: {'; '.join(detected_issues)}"
            )
        
        return PassResult()
