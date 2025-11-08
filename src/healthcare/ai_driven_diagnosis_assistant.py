import datetime
import json
import yaml
import os
import logging
import random
import asyncio
from collections import deque
from typing import Dict, Any, List
import redis


# Mock components
class MockFeatureStoreClient:
    def __init__(self, config_path, environment):
        self.client = redis.StrictRedis(
            host="localhost", port=6379, db=0, decode_responses=True
        )
        self.client.set(
            "patient_record:PAT_001",
            json.dumps(
                {
                    "age": 45,
                    "gender": "male",
                    "symptoms": ["cough", "fever"],
                    "medical_history": ["asthma"],
                    "admission_date": datetime.datetime.utcnow().isoformat() + "Z",
                }
            ),
        )
        self.client.set(
            "imaging_scan:SCAN_001",
            json.dumps(
                {
                    "patient_id": "PAT_001",
                    "scan_type": "xray_chest",
                    "status": "pending_analysis",
                    "scan_url": "mock_url/scan001.jpg",
                    "upload_time": datetime.datetime.utcnow().isoformat() + "Z",
                }
            ),
        )
        self.client.set(
            "lab_result:LAB_001",
            json.dumps(
                {
                    "patient_id": "PAT_001",
                    "test_type": "blood_panel",
                    "results": {"white_blood_cells": 11.5, "crp": 15.2},
                    "status": "finalized",
                }
            ),
        )
        self.client.set(
            "medical_guideline:PNEUMONIA_GUIDELINE",
            json.dumps(
                {
                    "diagnostic_criteria": {
                        "symptoms": ["cough", "fever", "shortness_of_breath"],
                        "imaging_findings": ["infiltrates"],
                        "lab_markers": {"crp_min": 10.0},
                    },
                    "recommended_treatments": ["antibiotics"],
                }
            ),
        )

    def get_feature(self, feature_group: str, key: str):
        data = self.client.get(f"{feature_group}:{key}")
        return json.loads(data) if data else {}

    def get_all_features_by_group(self, feature_group: str, pattern: str = "*"):
        keys = self.client.keys(f"{feature_group}:{pattern}")
        return (
            {k.split(":")[1]: json.loads(self.client.get(k)) for k in keys}
            if keys
            else {}
        )

    def set_feature(self, feature_group: str, key: str, value: Dict[str, Any]):
        self.client.set(f"{feature_group}:{key}", json.dumps(value))


class MockMedicalImagingAI:
    def analyze_xray(self, scan_url: str) -> Dict[str, Any]:
        if random.random() < 0.3:  # 30% chance of finding something
            findings = random.choice(
                ["lung_infiltrate", "cardiomegaly", "pleural_effusion"]
            )
            severity = random.uniform(0.5, 0.9)
            return {
                "status": "SUCCESS",
                "findings": [
                    {"type": findings, "severity": severity, "confidence": 0.9}
                ],
                "impression": "Findings suggestive of pneumonia.",
            }
        return {
            "status": "SUCCESS",
            "findings": [],
            "impression": "No significant findings.",
        }


class MockNLPClinicalAssistant:
    def extract_symptoms_and_context(self, clinical_notes: str) -> Dict[str, Any]:
        return {
            "symptoms": ["cough", "fever", "fatigue"],
            "duration_days": 3,
            "patient_description": "Patient reports general malaise.",
        }

    def identify_potential_conditions(
        self, patient_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        conditions = []
        if "fever" in patient_data.get("symptoms", []) and "cough" in patient_data.get(
            "symptoms", []
        ):
            conditions.append(
                {
                    "condition": "pneumonia",
                    "likelihood": random.uniform(0.4, 0.8),
                    "confidence": 0.7,
                }
            )
            conditions.append(
                {
                    "condition": "bronchitis",
                    "likelihood": random.uniform(0.3, 0.7),
                    "confidence": 0.6,
                }
            )
        return conditions


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIDrivenDiagnosisAssistant:
    def __init__(self, config_path="conf/environments/prod.yaml", environment="dev"):
        self.config = self._load_config(config_path)
        self.ai_diagnosis_config = self.config["environments"][environment][
            "ai_diagnosis_assistant"
        ]

        self.feature_store = MockFeatureStoreClient(config_path, environment)
        self.imaging_ai = MockMedicalImagingAI()
        self.nlp_assistant = MockNLPClinicalAssistant()

        self.pending_cases = deque()  # Patient IDs waiting for analysis
        logger.info("AIDrivenDiagnosisAssistant initialized.")

    def _load_config(self, config_path):
        if not os.path.exists(config_path):
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as f:
                f.write(
                    """
environments:
  dev:
    redis:
      host: localhost
      port: 6379
      db: 0
    ai_diagnosis_assistant:
      enabled: true
      analysis_interval_seconds: 10
      min_confidence_for_recommendation: 0.7
      diagnostic_guidelines_prefix: "medical_guideline:"
      default_treatment_recommendation: "Consult specialist for further evaluation."
"""
                )
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    async def _analyze_patient_case(self, patient_id: str):
        """Orchestrates the AI diagnosis process for a single patient."""
        logger.info(f"Analyzing patient case for {patient_id}...")
        patient_record = self.feature_store.get_feature("patient_record", patient_id)
        if not patient_record:
            logger.warning(f"Patient record {patient_id} not found.")
            return

        all_scans = self.feature_store.get_all_features_by_group(
            f"imaging_scan:SCAN_*:patient_id_{patient_id}"
        )  # Mock pattern
        all_lab_results = self.feature_store.get_all_features_by_group(
            f"lab_result:LAB_*:patient_id_{patient_id}"
        )  # Mock pattern

        # 1. Process imaging scans
        imaging_findings = []
        for scan_id, scan_data in all_scans.items():
            if (
                scan_data.get("status") == "pending_analysis"
                and scan_data.get("scan_type") == "xray_chest"
            ):
                logger.debug(f"Analyzing X-ray scan {scan_id} for {patient_id}...")
                ai_result = self.imaging_ai.analyze_xray(scan_data["scan_url"])
                if ai_result["status"] == "SUCCESS":
                    imaging_findings.extend(ai_result["findings"])
                    scan_data["ai_analysis_result"] = ai_result
                    scan_data["status"] = "analyzed"
                    self.feature_store.set_feature("imaging_scan", scan_id, scan_data)
                    logger.info(
                        f"AI imaging analysis for {patient_id} ({scan_id}) complete. Findings: {ai_result['findings']}"
                    )

        # 2. Integrate all data for NLP analysis
        combined_patient_data = {
            "patient_id": patient_id,
            "age": patient_record.get("age"),
            "gender": patient_record.get("gender"),
            "symptoms": patient_record.get("symptoms", []),
            "medical_history": patient_record.get("medical_history", []),
            "imaging_findings": imaging_findings,
            "lab_results": {
                k: v["results"]
                for k, v in all_lab_results.items()
                if v.get("status") == "finalized"
            },
        }

        # Mock NLP processing of symptoms from patient record
        potential_conditions = self.nlp_assistant.identify_potential_conditions(
            combined_patient_data
        )
        logger.debug(
            f"NLP identified potential conditions for {patient_id}: {potential_conditions}"
        )

        # 3. Apply medical guidelines for recommendations
        diagnosis_recommendations = []
        for condition_candidate in potential_conditions:
            if (
                condition_candidate["likelihood"]
                < self.ai_diagnosis_config["min_confidence_for_recommendation"]
            ):
                continue

            guideline_key = f"{self.ai_diagnosis_config['diagnostic_guidelines_prefix']}{condition_candidate['condition'].upper()}_GUIDELINE"
            guideline = self.feature_store.get_feature(
                "medical_guideline", guideline_key
            )

            if guideline:
                diagnostic_criteria = guideline.get("diagnostic_criteria", {})
                meets_criteria = True

                # Check symptoms
                for sym in diagnostic_criteria.get("symptoms", []):
                    if sym not in combined_patient_data["symptoms"]:
                        meets_criteria = False
                        break
                # Check imaging findings
                if meets_criteria:
                    required_imaging_findings = diagnostic_criteria.get(
                        "imaging_findings", []
                    )
                    if required_imaging_findings and not any(
                        f["type"] in required_imaging_findings for f in imaging_findings
                    ):
                        meets_criteria = False
                # Check lab markers
                if meets_criteria:
                    required_lab_markers = diagnostic_criteria.get("lab_markers", {})
                    for marker, min_val in required_lab_markers.items():
                        # Simplified, assumes specific lab_result keys
                        if not any(
                            lab_res.get(marker, 0) >= min_val
                            for lr_id, lab_res in combined_patient_data[
                                "lab_results"
                            ].items()
                        ):
                            meets_criteria = False
                            break

                if meets_criteria:
                    diagnosis_recommendations.append(
                        {
                            "condition": condition_candidate["condition"],
                            "likelihood": condition_candidate["likelihood"],
                            "treatment_recommendation": guideline.get(
                                "recommended_treatments", []
                            ),
                        }
                    )

        if not diagnosis_recommendations:
            diagnosis_recommendations.append(
                {
                    "condition": "Undetermined",
                    "likelihood": 0.0,
                    "treatment_recommendation": [
                        self.ai_diagnosis_config["default_treatment_recommendation"]
                    ],
                }
            )

        # Store comprehensive AI diagnosis in feature store
        ai_diagnosis_report = {
            "patient_id": patient_id,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "status": "completed",
            "imaging_summary": imaging_findings,
            "potential_conditions_nlp": potential_conditions,
            "diagnosis_recommendations": diagnosis_recommendations,
        }
        self.feature_store.set_feature(
            "ai_diagnosis_report", patient_id, ai_diagnosis_report
        )
        logger.info(
            f"AI diagnosis for {patient_id} complete. Recommendations: {diagnosis_recommendations}"
        )

    async def run_diagnosis_loop(self):
        """Main loop for continuously processing pending patient cases."""
        if not self.ai_diagnosis_config["enabled"]:
            logger.info("AI-Driven Diagnosis Assistant is disabled.")
            return

        while True:
            logger.info("Checking for new patient cases or pending analyses...")

            # Find patients with pending imaging scans or new patient records needing initial analysis
            new_or_pending_scans = self.feature_store.get_all_features_by_group(
                "imaging_scan", pattern="*"
            )
            patients_to_analyze = set()
            for scan_id, scan_data in new_or_pending_scans.items():
                if scan_data.get("status") == "pending_analysis":
                    patients_to_analyze.add(scan_data["patient_id"])

            # Also check if any new patient records are added that haven't been analyzed
            all_patients = self.feature_store.get_all_features_by_group(
                "patient_record"
            )
            for pat_id, pat_data in all_patients.items():
                if (
                    not self.feature_store.get_feature("ai_diagnosis_report", pat_id)
                    and pat_id not in patients_to_analyze
                ):
                    patients_to_analyze.add(pat_id)

            if not patients_to_analyze:
                logger.info("No new patient cases for AI diagnosis. Waiting...")
                await asyncio.sleep(
                    self.ai_diagnosis_config["analysis_interval_seconds"]
                )
                continue

            analysis_tasks = [
                self._analyze_patient_case(pid) for pid in patients_to_analyze
            ]
            await asyncio.gather(*analysis_tasks)

            await asyncio.sleep(self.ai_diagnosis_config["analysis_interval_seconds"])


if __name__ == "__main__":
    import redis

    config_file = "conf/environments/dev.yaml"
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    if not os.path.exists(config_file):
        with open(config_file, "w") as f:
            f.write(
                """
environments:
  dev:
    redis:
      host: localhost
      port: 6379
      db: 0
    ai_diagnosis_assistant:
      enabled: true
      analysis_interval_seconds: 5 # Faster for demo
      min_confidence_for_recommendation: 0.6
      diagnostic_guidelines_prefix: "medical_guideline:"
      default_treatment_recommendation: "Consult specialist for further evaluation."
"""
            )

    try:
        r = redis.StrictRedis(host="localhost", port=6379, db=0, decode_responses=True)
        r.ping()
        print(
            "Connected to Redis. Populating dummy data for AIDrivenDiagnosisAssistant."
        )

        # Patient with symptoms, pending xray, and lab results
        r.set(
            "patient_record:PAT_001",
            json.dumps(
                {
                    "patient_id": "PAT_001",
                    "age": 45,
                    "gender": "male",
                    "symptoms": ["cough", "fever", "shortness_of_breath"],
                    "medical_history": ["asthma"],
                    "admission_date": datetime.datetime.utcnow().isoformat() + "Z",
                }
            ),
        )
        r.set(
            "imaging_scan:SCAN_001",
            json.dumps(
                {
                    "patient_id": "PAT_001",
                    "scan_type": "xray_chest",
                    "status": "pending_analysis",
                    "scan_url": "mock_url/scan001.jpg",
                    "upload_time": datetime.datetime.utcnow().isoformat() + "Z",
                }
            ),
        )
        r.set(
            "lab_result:LAB_001",
            json.dumps(
                {
                    "patient_id": "PAT_001",
                    "test_type": "blood_panel",
                    "results": {"white_blood_cells": 12.0, "crp": 18.5},
                    "status": "finalized",
                }
            ),
        )

        # Medical guideline for pneumonia
        r.set(
            "medical_guideline:PNEUMONIA_GUIDELINE",
            json.dumps(
                {
                    "diagnostic_criteria": {
                        "symptoms": ["cough", "fever", "shortness_of_breath"],
                        "imaging_findings": ["lung_infiltrate"],
                        "lab_markers": {"crp": {"min_value": 10.0}},
                    },
                    "recommended_treatments": [
                        "antibiotics (e.g., Azithromycin)",
                        "rest",
                        "fluids",
                    ],
                }
            ),
        )

        # Another patient, no immediate pending scans, but needs initial check
        r.set(
            "patient_record:PAT_002",
            json.dumps(
                {
                    "patient_id": "PAT_002",
                    "age": 28,
                    "gender": "female",
                    "symptoms": ["headache", "fatigue"],
                    "medical_history": [],
                    "admission_date": (
                        datetime.datetime.utcnow() - datetime.timedelta(hours=1)
                    ).isoformat()
                    + "Z",
                }
            ),
        )

    except redis.exceptions.ConnectionError:
        print(
            "Redis not running. AI-Driven Diagnosis Assistant will start with empty data."
        )

    async def main_ai_diagnosis():
        assistant = AIDrivenDiagnosisAssistant(config_file)
        print("Starting AIDrivenDiagnosisAssistant for 30 seconds...")
        try:
            await asyncio.wait_for(assistant.run_diagnosis_loop(), timeout=30)
        except asyncio.TimeoutError:
            print("\nAIDrivenDiagnosisAssistant demo timed out after 30 seconds.")
        except KeyboardInterrupt:
            print("\nAIDrivenDiagnosisAssistant demo stopped by user.")

    asyncio.run(main_ai_diagnosis())
