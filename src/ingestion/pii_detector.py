from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer import PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from dataclasses import dataclass, field
from typing import List, Dict


# ── Custom recognizer for domain-specific PII ─────────────────────
# Presidio doesn't know about employee IDs in EMP-XXXXX format.
# We add it as a regex pattern with high confidence.
def build_employee_id_recognizer() -> PatternRecognizer:
    pattern = Pattern(
        name="employee_id_pattern",
        regex=r"\bEMP-\d{5}\b",  # matches EMP-12345
        score=0.95               # high confidence — very specific pattern
    )
    return PatternRecognizer(
        supported_entity="EMPLOYEE_ID",
        patterns=[pattern],
        name="EmployeeIdRecognizer"
    )


# ── Custom recognizer for salary information ──────────────────────
# Salaries like "$95,000" or "L4-$95,000" are sensitive compensation data.
def build_salary_recognizer() -> PatternRecognizer:
    patterns = [
        # Match dollar signs followed by 4+ digits (e.g., $10,000 or $95000), ignoring small amounts like $5
        Pattern("salary_usd", r"\$\b\d{1,3}(?:,\d{3})+\b|\$\b\d{4,7}\b", 0.8),
        # Keep your specific salary band format
        Pattern("salary_band", r"\bL\d[-–]\$[\d,]+", 0.9),
    ]
    return PatternRecognizer(
        supported_entity="SALARY",
        patterns=patterns,
        name="SalaryRecognizer"
    )


# ── PIIDetector class ─────────────────────────────────────────────
class PIIDetector:
    """
    Detects and anonymizes PII using Microsoft Presidio.
    Supports: PERSON, EMAIL, PHONE, EMPLOYEE_ID, SALARY,
              CREDIT_CARD, SSN, IP_ADDRESS, LOCATION, DATE_TIME
    """

    # Which entity types we care about for this project
    ENTITIES = [
        "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER",
        "CREDIT_CARD", "US_SSN", "IP_ADDRESS",
        "LOCATION", "DATE_TIME",
        "EMPLOYEE_ID", "SALARY",  # ← our custom recognizers
    ]

    def __init__(self):
        # Use the large spaCy model — better NER accuracy than sm/md
        nlp_config = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
        }
        provider = NlpEngineProvider(nlp_configuration=nlp_config)
        nlp_engine = provider.create_engine()

        # Build registry and add custom recognizers
        registry = RecognizerRegistry()
        registry.load_predefined_recognizers(nlp_engine=nlp_engine)
        registry.add_recognizer(build_employee_id_recognizer())
        registry.add_recognizer(build_salary_recognizer())

        self.analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine,
            registry=registry
        )
        self.anonymizer = AnonymizerEngine()

        print("✅ PIIDetector initialised with custom recognizers")

    def analyze(self, text: str) -> List:
        """Return list of detected PII spans with type, position, score."""
        return self.analyzer.analyze(
            text=text,
            entities=self.ENTITIES,
            language="en",
            score_threshold=0.6  # only report if >60% confident
        )

    def anonymize(self, text: str) -> Dict:
        """
        Returns dict with:
          - anonymized_text: text with PII replaced by [ENTITY_TYPE]
          - pii_found: list of detected entity types
          - pii_count: total number of PII items found
        """
        analyzer_results = self.analyze(text)

        if not analyzer_results:
            return {
                "anonymized_text": text,
                "pii_found": [],
                "pii_count": 0,
                "pii_detected": False
            }

        # Replace each PII span with [ENTITY_TYPE]
        # e.g. "Alice Chen" → "[PERSON]", "alice@co.com" → "[EMAIL_ADDRESS]"
        operators = {
            entity: OperatorConfig("replace", {"new_value": f"[{entity}]"})
            for entity in self.ENTITIES
        }

        anonymized = self.anonymizer.anonymize(
            text=text,
            analyzer_results=analyzer_results,
            operators=operators
        )

        entity_types = list({r.entity_type for r in analyzer_results})

        return {
            "anonymized_text": anonymized.text,
            "pii_found": entity_types,
            "pii_count": len(analyzer_results),
            "pii_detected": True
        }


# ── Quick test ────────────────────────────────────────────────────
if __name__ == "__main__":
    detector = PIIDetector()

    test_text = """
    Employee: Alice Chen (EMP-12345)
    Email: alice.chen@acmecorp.com | Phone: 555-867-5309
    Salary Band: L5-$115,000
    Manager: Bob Ramirez
    Review: Alice consistently exceeds expectations in Q4 2023.
    """

    result = detector.anonymize(test_text)
    print("Original:", test_text)
    print("Anonymized:", result["anonymized_text"])
    print("PII found:", result["pii_found"])
    print("PII count:", result["pii_count"])