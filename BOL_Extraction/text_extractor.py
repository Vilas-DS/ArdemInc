from PIL import Image as PILImage
from dataclasses import dataclass, field
import re
import os
import numpy as np
from typing import List, Optional
import configparser
import requests
import json
import logging

@dataclass
class BillPage:
    page_pil_image: PILImage.Image
    page_extracted_text: str
    page_number: str

@dataclass
class Bill:
    pages: List[BillPage] = field(default_factory=list)

class TextExtractor:
    def __init__(self, logger):
        self.logger = logger
        self.paddle_ocr = None

    def extract_text_lines_from_pil(self, pil_image: PILImage.Image) -> List[str]:
        """Extracts all text lines from a PIL image."""
        try:
            image_np = np.array(pil_image.convert('RGB'))
            result = self.paddle_ocr.ocr(image_np, cls=True)
            if result and result[0]:
                return [line[1][0] for line in result[0]]
        except Exception as e:
            self.logger.error(f"Error during OCR text extraction: {e}")
        return []

    def extract_page_number(self, text: str) -> Optional[str]:
        """Extracts the page number from a line of text."""
        match = re.search(r"Page\s*(\d+)", text, re.IGNORECASE)
        return match.group(1) if match else None

    def parse_bills_from_pages(self, page_images: List[PILImage.Image]) -> List[Bill]:
        """Splits a list of page images into logical bills, where each bill starts on page '1'."""
        bills: List[Bill] = []
        current_bill: Optional[Bill] = None

        for i, page_image in enumerate(page_images):
            extracted_texts = self.extract_text_lines_from_pil(page_image)
            full_page_text = "\n".join(extracted_texts)

            page_number = next((num for text in extracted_texts if (num := self.extract_page_number(text))), None)
            
            if page_number is None:
                page_number = f"auto_{i+1}"
                self.logger.warning(f"Could not find page number on image {i}. Assigning '{page_number}'.")

            if page_number == "1":
                if current_bill:
                    bills.append(current_bill)
                    self.logger.info(f"Completed bill with {len(current_bill.pages)} pages.")
                current_bill = Bill()

            if current_bill is None:
                self.logger.warning(f"Skipping page '{page_number}' since it appears before the first 'Page 1'.")
                continue

            bill_page = BillPage(page_image, full_page_text, page_number)
            current_bill.pages.append(bill_page)

        if current_bill:
            bills.append(current_bill)
            self.logger.info(f"Completed final bill with {len(current_bill.pages)} pages.")

        return bills

class JsonExtractor:
    def __init__(self, logger):
        self.logger = logger
        self.token = None
        self.model_prompt = """
        You are an expert parsing Bill of Lading (BOL) documents. Extract all relevant information from the following BOL text into a structured JSON format. If a field is not found, include it with an empty string or null. Provide only the JSON output.
        Example JSON:
        {"Ship From": {"Company Name": "", "Address": "", "Contact": "", "Phone": ""}, "Ship To": {"Company Name": "", "Address": "", "Contact": "", "Phone": ""}, "Carrier Information": {"Carrier Name": "", "SCAC": "", "Pro Number": ""}, "Bill of Lading Number": "", "Pickup Date": "", "Special Instructions": "", "Total Pieces": "", "Total Weight": "", "Payment Terms": ""}
        """

    def _load_credentials(self, config_path='config.ini'):
        config = configparser.ConfigParser()
        if not os.path.exists(config_path):
            self.logger.error(f"Configuration file not found at {config_path}")
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        config.read(config_path)
        return config.get('credentials', 'username'), config.get('credentials', 'password')

    def _generate_bearer_token(self):
        if self.token: return self.token
        try:
            user, pwd = self._load_credentials()
            response = requests.post("https://transportationllmbackend1.ardemcloud.dev/token", data={"username": user, "password": pwd})
            response.raise_for_status()
            self.token = response.json().get("access_token")
            self.logger.info("Successfully generated new bearer token.")
            return self.token
        except Exception as e:
            self.logger.error(f"Failed to generate bearer token: {e}")
            return None

    def get_json_data(self, extracted_text: str) -> dict:
        """Sends extracted text to an LLM API and returns the structured JSON response."""
        token = self._generate_bearer_token()
        if not token: return {}

        llm_url = "https://transportationllmbackend1.ardemcloud.dev/api_LLM_process/v1"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {"extracted_text": extracted_text, "model_prompt": self.model_prompt}

        try:
            response = requests.post(llm_url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            if not result:
                self.logger.warning("LLM response is empty.")
                return {}
            output_content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            if not output_content:
                self.logger.warning("LLM response content is empty.")
                return {}
            
            if '```json' in output_content:
                match = re.search(r'```json\s*([\s\S]*?)\s*```', output_content)
                if match:
                    output_content = match.group(1)
            
            return json.loads(output_content)

        except requests.exceptions.RequestException as e:
            self.logger.error(f"LLM API request failed: {e}")
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to decode JSON from LLM response: {e}\nResponse text: {output_content}")
        except (KeyError, IndexError, TypeError, AttributeError) as e:
            self.logger.error(f"Error parsing LLM response structure: {e}\nResponse: {result}")
        
        return {}