import warnings
warnings.filterwarnings('ignore')

import os
import logging
from typing import List, Tuple, Dict, Any, Optional, Callable

from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from paddleocr import PaddleOCR
from PIL import Image  # noqa: F401

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

        self.paddle_ocr: Optional[PaddleOCR] = None
        self.table_detector = None
        self.table_detector_processor = None
        self._initialized = False

    # ---------------- init ----------------

    def initialize(self):
        """Initializes OCR and TATR models once and injects them into dependencies."""
        if self._initialized:
            return

        try:
            self.logger.info("Initializing PaddleOCR...")
            self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log = False) 
            self.text_extractor.paddle_ocr = self.paddle_ocr
            self.table_extractor.paddle_ocr = self.paddle_ocr

            self.logger.info("Initializing TATR detection models...")
            self.table_detector, self.table_detector_processor = self.table_extractor.initialize_tatr_detection_models()
            self.table_extractor.table_detector = self.table_detector
            self.table_extractor.table_detector_processor = self.table_detector_processor

            self._initialized = True
            self.logger.info("Core extraction models initialized successfully.")
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {e}")
            raise

    # ---------------- helpers ----------------

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

    def _process_dataframes(self, dfs: List[pd.DataFrame]) -> Tuple[list, list, dict]:
        """
        Processes a list of DataFrames to extract structured information and totals.
        Returns (customer_order_info, carrier_info, totals).
        """
        customer_order_info, carrier_info = [], []
        totals = {'total_packages': "", 'total_weight': "", 'handling_units_qty': ""}

        for df in dfs:
            df_keys_lowered = [str(col).lower().strip() for col in df.columns]

            # Customer order info & totals
            if any(any(text in col for col in df_keys_lowered) for text in _CUSTOMER_ORDER_KEYWORDS):
                records = [
                    rec for rec in df.to_dict(orient="records")
                    if "GRAND TOTAL" not in str(rec.values())
                ]
                customer_order_info.extend(records)

                total_dict = next(
                    (rec for rec in df.to_dict(orient="records") if "GRAND TOTAL" in str(rec.values())), {}
                )
                pkg_key = next(
                    (k for k in total_dict if any(text in k.lower() for text in ['#pkgs', '# pkgs', 'package'])),
                    None
                )
                if pkg_key:
                    totals['total_packages'] = total_dict.get(pkg_key, "")
                weight_key = next((k for k in total_dict if 'weight' in k.lower()), None)
                if weight_key:
                    totals['total_weight'] = total_dict.get(weight_key, "")

            # Carrier info & totals
            elif any(any(text in col for col in df_keys_lowered) for text in _CARRIER_INFO_KEYWORDS):
                records = [
                    rec for rec in df.to_dict(orient="records")
                    if "GRAND TOTAL" not in str(rec.values())
                ]
                carrier_info.extend(records)

                total_dict = next(
                    (rec for rec in df.to_dict(orient="records") if "GRAND TOTAL" in str(rec.values())), {}
                )
                handling_qty_key = next(
                    (k for k in total_dict if 'handling' in k.lower() and 'qty' in k.lower()),
                    None
                )
                if handling_qty_key:
                    totals['handling_units_qty'] = total_dict.get(handling_qty_key, "")

        return customer_order_info, carrier_info, totals

    def _emit(self, cb: Optional[Callable[[Dict[str, Any]], None]], payload: Dict[str, Any]):
        if cb is not None:
            try:
                cb(payload)
            except Exception:
                # Don't let progress reporting affect extraction
                pass

    # ---------------- per-bill extraction ----------------

    def _save_page_image(
        self,
        bill_index: int,
        rel_page_in_bill: int,
        abs_page_number: int,
        pil_image: "Image.Image",
        result_id: str,
        output_dir: str
    ) -> str:
        """
        Save page image under a bill-scoped subfolder to avoid any cross-bill collisions:
            /static/results/{result_id}/Bill_{bill_index}/page_{rel_page_in_bill}.png

        Returns the public URL starting at /static/...
        """
        bill_dir = os.path.join(output_dir, f"Bill_{bill_index}")
        os.makedirs(bill_dir, exist_ok=True)

        filename = f"page_{rel_page_in_bill}.png"
        path = os.path.join(bill_dir, filename)
        pil_image.save(path, "PNG")

        url = f"/static/results/{result_id}/Bill_{bill_index}/{filename}"
        return url


    def transform_to_req_schema(self, old_json: dict) -> dict:
        """
        Transform the provided old JSON schema (with Ship From/Ship To/Freight Description/etc.)
        into the new target schema with Carrier, BillTo, Freight Charge Terms, etc.
        """

        # Extract top-level old fields
        ship_from = old_json.get("Ship From", {})
        ship_to = old_json.get("Ship To", {})
        bill_to = old_json.get("BillTo", {})
        carrier_info = old_json.get("Carrier Information", {})
        customer_orders = old_json.get("Customer Order Information", {})
        freight_desc = old_json.get("Freight Description", [])


        # CustomerPO = first or all order numbers
        customer_po = ""
        if customer_orders:
            customer_po = ", ".join(
                co.get("CUSTOMER ORDER NUMBER", "") for co in customer_orders if co.get("CUSTOMER ORDER NUMBER")
            )

        # LineItems from Carrier Information
        line_items = []
        for ci in carrier_info:
            line_items.append({
                "NMFC": ci.get("(LTL ONLY) NMFC #", ""),
                "Class": ci.get("(LTL ONLY) CLASS", ""),
                "Pieces": ci.get("PACKAGE QTY", ""),
                "Description": ci.get("COMMODITY DESCRIPTION", ""),
                "Weight": ci.get("WEIGHT (LB)", ""),
            })

        # Map to new schema
        new_json = {
            "PRO": old_json.get("Pro Number", ""),
            "BOLNumber": old_json.get("Bill of Lading Number", ""),
            "Payment": old_json.get("Payment Terms", ""),
            "Shipper": {
                "Name": ship_from.get("Company Name", ""),
                "Street": ship_from.get("Address", ""),
                "CityStateZip": ship_from.get("CityStateZip", ""),   # Could parse from Address if needed
                "Phone": ship_from.get("Phone", ""),
                "Contact": ship_from.get("Contact", ""),
                "ReceivingHours": ""
            },
            "Consignee": {
                "Name": ship_to.get("Company Name", ""),
                "Street": ship_to.get("Address", ""),
                "CityStateZip": ship_to.get("CityStateZip", ""),   # Same as above
                "Phone": ship_to.get("Phone", ""),
                "Contact": ship_to.get("Contact", ""),
                "ReceivingHours": ""
            },
            "BillTo": old_json.get("BillTo", {"Name": "",
                                            "Street": "",
                                            "CityStateZip": "",
                                            "Contact": ""}
                                            ),
            "CustomerPO": customer_po,
            "LineItems": line_items,
            "PickupDetails": {
                "PickupID": "",
                "PickupDate": old_json.get("Pickup Date", ""),
                "ShipDate": "",
                "MPO": "",
                "MRF": "",
                "MQuote": "",
                "MBefore": "",
                "MDim": ", ".join([item.get("Dimensions", "") for item in freight_desc if item.get("Dimensions")]),
                "MM": ""
            }
        }

        return new_json


    def get_bill_data(
        self,
        bill_index: int,
        bill_pages: List[Any],
        result_id: str,
        output_dir: str,
        progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Tuple[List[pd.DataFrame], str, List[str], List[Dict[str, Any]]]:
        """
        Extracts data and saves page images for a single bill, with explicit bill scoping.

        Returns:
            all_dfs:            list[pd.DataFrame] for ALL tables across this bill (aggregate)
            bill_extracted_text: concatenated OCR text for this bill
            bill_page_image_urls: list[str] image URLs in bill-relative page order
            pages_info:         list[ { 'absolute_page': int,
                                        'relative_page': int,
                                        'image_url': str,
                                        'tables': list[list-of-row-dicts] } ]
        """
        all_dfs: List[pd.DataFrame] = []
        bill_extracted_text = ""
        bill_page_image_urls: List[str] = []
        pages_info: List[Dict[str, Any]] = []

        for rel_idx, bill_page in enumerate(bill_pages, start=1):
            abs_page_number = int(getattr(bill_page, "page_number", rel_idx))
            pil_image = bill_page.page_pil_image
            page_text = bill_page.page_extracted_text or ""

            # Append page text
            bill_extracted_text += f"Page_{abs_page_number}:\n{page_text}\n\n"

            # Save image into Bill_{i}/page_{rel}.png
            try:
                img_url = self._save_page_image(
                    bill_index=bill_index,
                    rel_page_in_bill=rel_idx,
                    abs_page_number=abs_page_number,
                    pil_image=pil_image,
                    result_id=result_id,
                    output_dir=output_dir
                )
                bill_page_image_urls.append(img_url)
            except Exception as e:
                self.logger.error(f"Failed to save image for Bill {bill_index} page {abs_page_number}: {e}")
                img_url = ""

            # ---- Detect/extract tables on THIS page only ----
            page_tables: List[pd.DataFrame] = []

            parts = self.table_extractor.split_image_by_black_strips(
                pil_image,
                strip_height_threshold=2,
                black_pixel_ratio=0.6
            )
            self.logger.info(f"[Bill {bill_index}] AbsPage {abs_page_number} split into {len(parts)} segment(s).")

            for part_img in parts:
                tables = self.table_extractor.detect_tables(part_img, threshold=0.95)
                if not tables:
                    continue

                for t_i, table_box in enumerate(tables, start=1):
                    try:
                        logging.info(f"Extracting table {t_i} on abs page {abs_page_number} (Bill {bill_index})")
                        table_image = self.table_extractor.crop_table_image(part_img, table_box, max_size=1600, padding=20)

                        row_segs = self.table_extractor.crop_rows_from_horizontal_lines(
                            table_image, width_req_ratio=0.1, crop_only_first=True
                        )
                        if not row_segs or len(row_segs) < 2:
                            self.logger.info(f"Skip table on abs page {abs_page_number}: cannot segment header/body.")
                            continue

                        header_body = row_segs[0]
                        table_body = row_segs[-1]
                        col_segs_all = self.table_extractor.crop_columns_from_vertical_lines(
                            table_image, height_req_ratio=0.1, min_col_width=20
                        )
                        col_segs_body_only = self.table_extractor.crop_columns_from_vertical_lines(
                            table_body, min_col_width=20
                        )

                        headers, col_segs_body_only = self.table_extractor.get_headers(
                            header_body, table_body, col_segs_all
                        )
                        df = self.table_extractor.extract_df(col_segs_body_only, headers)
                        df = df.replace('', np.nan).dropna(how='all').fillna('')

                        if self._is_dummy_table(df):
                            self.logger.info(f"Dummy table on abs page {abs_page_number} (Bill {bill_index}), skipping.")
                            continue

                        page_tables.append(df)
                        all_dfs.append(df)
                    except Exception as e:
                        self.logger.warning(f"Error During Processing Table Image: {e}")

            pages_info.append({
                "absolute_page": abs_page_number,
                "relative_page": rel_idx,
                "image_url": img_url,
                "tables": [df.to_dict(orient='records') for df in page_tables]
            })

            # Emit progress tick per completed page
            self._emit(progress_cb, {"event": "tick", "delta": 1})

        return all_dfs, bill_extracted_text, bill_page_image_urls, pages_info

    # ---------------- document-level entrypoint ----------------

    def extract_json_data_tables(
        self,
        pdf_path: str,
        result_id: str,
        output_dir: str,
        progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Main method to extract all data and save images to the provided directory.

        Returns:
            {
              "data": {
                 "Bill_1": {
                    "json_data": {...},
                    "data_tables": [ ... all tables across the bill ... ],
                    "page_images": [ "/static/results/{id}/Bill_1/page_1.png", ... ],
                    "pages": [
                       { "absolute_page": 10,
                         "relative_page": 1,
                         "image_url": ".../Bill_1/page_1.png",
                         "tables": [ [...], ... ] },
                       ...
                    ]
                 },
                 ...
              },
              "meta": { "pages": <int>, "bills": <int> }
            }
        """
        self.initialize()

        # Convert PDF -> images; order preserved and absolute page numbers set by TableExtractor
        doc_pages = self.table_extractor.pdf_to_images(pdf_path, max_size=1600, DPI=240)
        bills = self.text_extractor.parse_bills_from_pages(doc_pages)

        total_pages = len(doc_pages)
        total_bills = len(bills)
        self._emit(progress_cb, {"event": "meta", "pages": total_pages, "bills": total_bills})

        pdf_bills_data: Dict[str, Any] = {}

        for bill_num, bill in enumerate(bills, start=1):
            all_dfs, bill_text, page_urls, pages_info = self.get_bill_data(
                bill_index=bill_num,
                bill_pages=bill.pages,
                result_id=result_id,
                output_dir=output_dir,
                progress_cb=progress_cb
            )

            # Structured JSON extracted from text
            json_data = self.json_extractor.get_json_data(bill_text)

            # Domain-specific aggregations from tables
            customer_order_info, carrier_info, totals = self._process_dataframes(all_dfs)
            json_data['Total Packages'] = totals['total_packages']
            json_data['Total Weight'] = totals['total_weight']
            json_data['Total Handling Units'] = totals['handling_units_qty']
            json_data['Customer Order Information'] = customer_order_info
            json_data['Carrier Information'] = carrier_info

            bill_key = f"Bill_{bill_num}"
            pdf_bills_data[bill_key] = {
                "required_json": self.transform_to_req_schema(json_data),
                "json_data": json_data,
                "data_tables": [df.to_dict(orient='records') for df in all_dfs],  # legacy aggregate
                "page_images": page_urls,  # bill-scoped URLs now
                "pages": pages_info         # page-aligned mapping (use this for perfect alignment)
            }

        return {
            "data": pdf_bills_data,
            "meta": {"pages": total_pages, "bills": total_bills}
        }
