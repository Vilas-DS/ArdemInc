import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from PIL import Image
import os
import logging
import numpy as np
from paddleocr import PaddleOCR

# Constants for identifying table types
_CUSTOMER_ORDER_KEYWORDS = ['customer order number', '# pkgs', '#pkgs', 'additional shipper info']
_CARRIER_INFO_KEYWORDS = ['commodity description', 'handling unit', 'commodities requiring special']
_DUMMY_TABLE_KEYWORDS = ["see", "attached", "supplement", "page"]

class PDFExtractor:
    """Core PDF extraction logic wrapper."""

    def __init__(self, logger, text_extractor, table_extractor, json_extractor):
        self.logger = logger
        self.text_extractor = text_extractor
        self.table_extractor = table_extractor
        self.json_extractor = json_extractor
        
        self.paddle_ocr = None
        self.table_detector = None
        self.table_detector_processor = None
        self._initialized = False
        
    def initialize(self):
        """Initializes OCR and TATR models once and injects them into dependencies."""
        if self._initialized:
            return

        try:
            self.logger.info("Initializing PaddleOCR...")
            self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            self.text_extractor.paddle_ocr = self.paddle_ocr
            self.table_extractor.paddle_ocr = self.paddle_ocr

            self.logger.info("Initializing TATR detection models...")
            self.table_detector, self.table_detector_processor = self.table_extractor.initialize_tatr_detection_models()
            self.table_extractor.table_detector = self.table_detector
            self.table_extractor.table_detector_processor = self.table_detector_processor
            
            self.logger.info("TATR models initialized successfully.")
            self._initialized = True
            self.logger.info("Core extraction models initialized successfully.")

        except Exception as e:
            self.logger.error(f"Failed to initialize models: {e}")
            raise

    def _is_dummy_table(self, df: pd.DataFrame) -> bool:
        """Checks if a DataFrame represents a non-data table (e.g., a note)."""
        for _, row in df.iterrows():
            row_text = "".join(
                str(cell).strip().replace(' ', '').lower()
                for cell in row.values if isinstance(cell, str) and cell.strip()
            )
            if sum(1 for keyword in _DUMMY_TABLE_KEYWORDS if keyword in row_text) > 2:
                self.logger.warning("Dummy table found, skipping.")
                return True
        return False

    def _process_dataframes(self, dfs: list):
        """Processes a list of DataFrames to extract structured information and totals."""
        customer_order_info, carrier_info = [], []
        totals = {'total_packages': "", 'total_weight': "", 'handling_units_qty': ""}

        for df in dfs:
            df_keys_lowered = [str(col).lower().strip() for col in df.columns]
            
            if any(any(text in col for col in df_keys_lowered) for text in _CUSTOMER_ORDER_KEYWORDS):
                customer_order_info.extend([rec for rec in df.to_dict(orient="records") if "GRAND TOTAL" not in str(rec.values())])
                total_dict = next((rec for rec in df.to_dict(orient="records") if "GRAND TOTAL" in str(rec.values())), {})
                
                pkg_key = next((k for k in total_dict if any(text in k.lower() for text in ['#pkgs', '# pkgs', 'package'])), None)
                if pkg_key: totals['total_packages'] = total_dict.get(pkg_key, "")
                
                weight_key = next((k for k in total_dict if 'weight' in k.lower()), None)
                if weight_key: totals['total_weight'] = total_dict.get(weight_key, "")

            elif any(any(text in col for col in df_keys_lowered) for text in _CARRIER_INFO_KEYWORDS):
                carrier_info.extend([rec for rec in df.to_dict(orient="records") if "GRAND TOTAL" not in str(rec.values())])
                total_dict = next((rec for rec in df.to_dict(orient="records") if "GRAND TOTAL" in str(rec.values())), {})

                handling_qty_key = next((k for k in total_dict if 'handling' in k.lower() and 'qty' in k.lower()), None)
                if handling_qty_key: totals['handling_units_qty'] = total_dict.get(handling_qty_key, "")

        return customer_order_info, carrier_info, totals

    def get_bill_data(self, bill_pages, result_id, output_dir):
        """
        Extracts data and saves page images for a single bill.
        This version finds all tables on a page and processes them individually.
        """
        all_dfs = []
        bill_extracted_text = ""
        page_image_urls = [] 

        for bill_page in bill_pages:
            page_number = bill_page.page_number
            pil_image = bill_page.page_pil_image
            
            page_extracted_text  = bill_page.page_extracted_text
            bill_extracted_text += f"Page_{page_number}:\n{page_extracted_text}\n\n"

            try:
                img_filename = f"page_{page_number}.png"
                img_path = os.path.join(output_dir, img_filename)
                pil_image.save(img_path, "PNG")
                img_url = f"/static/results/{result_id}/{img_filename}"
                page_image_urls.append(img_url)
            except Exception as e:
                self.logger.error(f"Failed to save page image {page_number}: {e}")
            
            # Split image
            parts = self.table_extractor.split_image_by_black_strips(pil_image, strip_height_threshold=5, black_pixel_ratio=0.6)
            print(f"Length of Segments: {len(parts)}")
            for image in parts:
                tables = self.table_extractor.detect_tables(image, threshold=0.9)
                if not tables:
                    self.logger.info(f"No tables found on Page {page_number}.")
                    continue
                
                self.logger.info(f"Found {len(tables)} table(s) on Page {page_number}.")

                for i, table_box in enumerate(tables):
                    logging.info(f"Extracting table {i + 1}...")
                    table_image = self.table_extractor.crop_table_image(image, table_box, max_size=1600, padding=20)
                    row_segments_only_first = self.table_extractor.crop_rows_from_horizontal_lines(table_image, width_req_ratio=0.1, crop_only_first=True)
                    table_body = row_segments_only_first[-1]
                    header_body = row_segments_only_first[0]
                    col_segments = self.table_extractor.crop_columns_from_vertical_lines(table_image, height_req_ratio=0.1, min_col_width=20)
                    col_segments_body_only = self.table_extractor.crop_columns_from_vertical_lines(table_body, min_col_width=20)
                    headers, col_segments_body_only = self.table_extractor.get_headers(header_body, table_body, col_segments)
                    df = self.table_extractor.extract_df(col_segments_body_only, headers)
                    df = df.replace('', np.nan).dropna(how='all').fillna('')
                    dummy_df = False
                
                    for _, row in df.iterrows():
                        # Combine all non-empty cell text, remove spaces, and lowercase
                        row_text = "".join(
                            str(cell).strip().replace(' ', '').lower()
                            for cell in row.values
                            if isinstance(cell, str) and cell.strip()
                        )
                
                        # Check if all required keywords are present
                        check_ = [keyword in row_text for keyword in ["see", "attached", "supplement", "page"]]
                        if len([check for check in check_ if check]) > 2:
                            dummy_df = True
                            break
                
                    if dummy_df:
                        print("Dummy Table Found, Skipping........")      
                        continue
                    all_dfs.append(df)    
        
        return all_dfs, bill_extracted_text, page_image_urls

    def extract_json_data_tables(self, pdf_path: str, result_id: str, output_dir: str) -> dict:
        """Main method to extract all data and save images to the provided directory."""
        self.initialize()
        
        page_images = self.table_extractor.pdf_to_images(pdf_path, max_size=1600, DPI=360)
        bills = self.text_extractor.parse_bills_from_pages(page_images)
        
        pdf_bills_data = {}
        for bill_num, bill in enumerate(bills, start=1):
            dfs, bill_extracted_text, page_images = self.get_bill_data(bill.pages, result_id, output_dir)
            
            json_data = self.json_extractor.get_json_data(bill_extracted_text)
            
            customer_order_info, carrier_info, totals = self._process_dataframes(dfs)

            json_data['Total Packages'] = totals['total_packages']
            json_data['Total Weight'] = totals['total_weight']
            json_data['Total Handling Units'] = totals['handling_units_qty']
            json_data['Customer Order Information'] = customer_order_info
            json_data['Carrier Information'] = carrier_info

            pdf_bills_data[f"Bill_{bill_num}"] = {
                "json_data": json_data,
                "data_tables": [df.to_dict(orient='records') for df in dfs],
                "page_images": page_images
            }

        return pdf_bills_data