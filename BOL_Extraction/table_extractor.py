from typing import List, Tuple
from PIL import Image as PILImage
import pandas as pd
import fitz  # PyMuPDF
import torch
import cv2
import numpy as np
from transformers import TableTransformerForObjectDetection, AutoImageProcessor
import logging

class TableExtractor:
    def __init__(self, logger):
        self.logger = logger
        self.paddle_ocr = None
        self.table_detector = None
        self.table_detector_processor = None


    def pdf_to_images(self, pdf_path, max_size = 1600, DPI = 360, preprocess = False):
        """Converts a PDF file to a list of PIL Images using PyMuPDF."""
        print(f"Converting PDF '{pdf_path}' to images with DPI={DPI}...")
        try:
            doc = fitz.open(pdf_path)
            images = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=DPI)
                img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
                w, h = img.size
                scale = min(max_size / w, max_size / h, 1.0)  # Only shrink, don't enlarge
                new_size = (int(w * scale), int(h * scale))
                image_resized = img.resize(new_size, PILImage.LANCZOS)
                
                images.append(image_resized)
            doc.close()
            print(f"Successfully converted {len(images)} pages.")
            return images
        except Exception as e:
            logging.error(f"Error converting PDF: {e}")
            return []
        

    def split_image_by_black_strips(self, pil_image, strip_height_threshold=3, black_pixel_ratio=0.95):
        """
        Splits a PIL.Image vertically at each horizontal black strip (≥ strip_height_threshold thick).
        
        Args:
            pil_image (PIL.Image): Input image.
            strip_height_threshold (int): Minimum thickness of a black strip in pixels.
            black_pixel_ratio (float): Ratio of black pixels per row to consider it part of a black strip.
        
        Returns:
            List[PIL.Image]: List of cropped image parts (≥1).
        """
        gray = pil_image.convert('L')
        binary = np.array(gray) < 30  # Threshold to binary: True = black
        height, width = binary.shape

        # Step 1: Compute black pixel ratio per row
        row_black_ratios = binary.sum(axis=1) / width

        # Step 2: Detect contiguous black row regions as strips
        in_strip = False
        strip_start = None
        strip_bounds = []

        for i, ratio in enumerate(row_black_ratios):
            if ratio > black_pixel_ratio:
                if not in_strip:
                    strip_start = i
                    in_strip = True
            else:
                if in_strip:
                    strip_end = i
                    if (strip_end - strip_start) >= strip_height_threshold:
                        strip_bounds.append((strip_start, strip_end))
                    in_strip = False
        if in_strip and (height - strip_start) >= strip_height_threshold:
            strip_bounds.append((strip_start, height))

        # Step 3: Use strip bounds to split image
        if not strip_bounds:
            return [pil_image]  # No strip found

        segments = []
        prev_end = 0

        for start, end in strip_bounds:
            # Image above the strip
            if start > prev_end:
                segments.append(pil_image.crop((0, prev_end, width, start)))
            prev_end = end

        # Image below last strip
        if prev_end < height:
            segments.append(pil_image.crop((0, prev_end, width, height)))

        return segments
    
    def initialize_tatr_detection_models(self, use_fast = True):
        print("Initializing TATR detection Models")
        table_detector = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection", revision="no_timm")
        table_detector_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection", revision="no_timm", use_fast = use_fast)
        return table_detector, table_detector_processor

    def detect_tables(self, image: PILImage.Image,threshold = 0.9):
        
        inputs = self.table_detector_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.table_detector(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        table_results = self.table_detector_processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0]
        tables = [box for label, box in zip(table_results['labels'], table_results['boxes']) if self.table_detector.config.id2label[label.item()] == 'table']
        return tables
    
    def crop_table_image(self, original_image: PILImage.Image, table_box, max_size=1600, padding=10) -> PILImage.Image:
        table_box = np.array(table_box, dtype=np.float32)

        # Case 1: axis-aligned bounding box (x_min, y_min, x_max, y_max)
        if table_box.shape == (4,):
            x_min, y_min, x_max, y_max = map(int, table_box)
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(original_image.width, x_max + padding)
            y_max = min(original_image.height, y_max + padding)
            cropped = original_image.crop((x_min, y_min, x_max, y_max))

        # Case 2: quadrilateral for warped table
        elif table_box.shape == (4, 2):
            cropped = self.perspective_transform(original_image, table_box)

        # Handle error
        else:
            raise ValueError(f"Invalid table_box format: shape {table_box.shape}, data: {table_box}")

        # Resize while preserving aspect ratio
        w, h = cropped.size
        scale = min(max_size / w, max_size / h, 1.0)
        new_size = (int(w * scale), int(h * scale))
        return cropped.resize(new_size, PILImage.LANCZOS)

    def crop_rows_from_horizontal_lines(self, table_image, width_req_ratio=0.10, min_line_gap=20, crop_only_first = False):
        original_cv_image = np.array(table_image.convert('RGB'))
        # Convert to grayscale for processing
        gray_image = cv2.cvtColor(original_cv_image, cv2.COLOR_BGR2GRAY)
        # THRESH_BINARY_INV makes the lines white and the background black.
        thresh_image = cv2.adaptiveThreshold(
            ~gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2
        )
        # --- Morphological Operations to detect horizontal lines ---
        # Create a long horizontal kernel. The length should be a significant fraction of the image width.
        img_height, img_width = thresh_image.shape
        horizontal_kernel_length = int(np.ceil(img_width *width_req_ratio)) # This value can be tuned.
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_length, 1))
        # Apply morphological opening to remove small noise and text
        horizontal_opened = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, horizontal_kernel)

        # Dilate the lines to connect any breaks
        dilated_horizontal = cv2.dilate(horizontal_opened, horizontal_kernel, iterations=1)

        # Find contours of the lines and filter them    
        contours, _ = cv2.findContours(dilated_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    
        # Filter contours based on the width constraint (>80% of image width)
        min_line_width = int(img_width * 0.80)
        line_y_coords = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > min_line_width:
                # We use the y-coordinate of the bounding box as the line position
                line_y_coords.append(y)
        

        if not line_y_coords:
            return [table_image]  # fallback to whole image

        # Sort the y-coordinates from top to bottom
        line_y_coords.sort()

        line_bounds = []
        current_start = line_y_coords[0]

        for y in line_y_coords[1:]:
            if y - current_start > min_line_gap:
                line_bounds.append(current_start)
                current_start = y
        line_bounds.append(current_start)
        row_positions = [0] + line_bounds + [img_height]

        new_row_positions = [
            y1 for y1, y2 in zip(row_positions, row_positions[1:])
            if (y2 - y1) > min_line_gap
        ]
        # Always add the last row (bottom of image)
        new_row_positions.append(row_positions[-1])


        segments = []
        if crop_only_first:
            # first row = header
            segments.append(table_image.crop((0, new_row_positions[0], table_image.width, new_row_positions[1])))
            # rest = body
            segments.append(table_image.crop((0, new_row_positions[1], table_image.width, table_image.height)))

        else:
            for i in range(len(new_row_positions) - 1):
                y1, y2 = new_row_positions[i], new_row_positions[i + 1]
                crop = table_image.crop((0, y1, table_image.width, y2))
                segments.append(crop)
        return segments

    def crop_columns_from_vertical_lines(self, table_image, height_req_ratio=0.10, min_col_gap=5, min_col_width=20):
        original_cv_image = np.array(table_image.convert('RGB'))
        try:
            gray_image = cv2.cvtColor(original_cv_image, cv2.COLOR_BGR2GRAY)
        except:
            print("Empty Image, Skipping.....")
            return None

        thresh_image = cv2.adaptiveThreshold(
            ~gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2
        )

        img_height, img_width = thresh_image.shape
        vertical_kernel_length = int(np.ceil(img_height * height_req_ratio))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_length))

        vertical_opened = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, vertical_kernel)
        dilated_vertical = cv2.dilate(vertical_opened, vertical_kernel, iterations=1)

        contours, _ = cv2.findContours(dilated_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_line_height = int(img_height * 0.80)
        line_x_coords = [x for cnt in contours if cv2.boundingRect(cnt)[3] > min_line_height
                        for x, _, _, _ in [cv2.boundingRect(cnt)]]

        if not line_x_coords:
            print("Warning: No vertical lines meeting the criteria were found.")
            return [table_image]

        line_x_coords.sort()

        # Group lines based on min_col_gap
        grouped_lines = [line_x_coords[0]]
        for x in line_x_coords[1:]:
            if x - grouped_lines[-1] > min_col_gap:
                grouped_lines.append(x)

        col_positions = [0] + grouped_lines + [img_width]

        segments = []
        for i in range(len(col_positions) - 1):
            x1, x2 = col_positions[i], col_positions[i + 1]
            x2 = min(x2, table_image.width)
            if x2 - x1 > min_col_width:
                crop = table_image.crop((x1, 0, x2, table_image.height))
                segments.append(crop)

        return segments

    def get_headers(self, header_body, table_body, col_segments):
        header_img_array = np.array(header_body)
        results_header_body = self.paddle_ocr.ocr(header_img_array, cls=True)
        header_texts = []
        for line in results_header_body:
            for box, (text, confidence) in line:
                header_texts.append(text.lower().strip())

        if any(text in header_texts for text in ['commodity description', 'handling unit', 'commodilies requiring special']):
            headers = ['HANDLING UNITS QTY','HANDLING UNITS TYPE','PACKAGE QTY','PACKAGE TYPE', 'WEIGHT (LB)', 'H.M (X)',
                    'COMMODITY DESCRIPTION',
                    '(LTL ONLY) NMFC #', '(LTL ONLY) CLASS']
            
        elif any(text in header_texts for text in ['customer order number', '# pkgs', 'additional shipper info']):  
            headers = ['CUSTOMER ORDER NUMBER', '#PKGS', 'WEIGHT (LB)', 'Pallet (Circle One)', 'Slip (Circle One)',
                    'ADDITIONAL SHIPPER INFO']  

        else:
            headers = []
            for line in results_header_body:
                for box, (text, confidence) in line:
                    headers.append(text.strip())

        col_segments_body_only = self.crop_columns_from_vertical_lines(table_body, min_col_width=20)
        if not col_segments_body_only:
            raise Exception("Empty Image in Table Body")
        if len(headers) != len(col_segments_body_only):
            headers = []
            for col_seg in col_segments:
                cells = self.crop_rows_from_horizontal_lines(col_seg, width_req_ratio=0.9)
                header_cell = cells[0]
                header_cell_results = self.paddle_ocr.ocr(np.array(header_cell), cls=True)
                
                
                header_cell_texts = [text for line in header_cell_results for box, (text, confidence) in line]
                header_text = " ".join(header_cell_texts)
                headers.append(header_text)

            return headers, col_segments_body_only    
        
        return headers, col_segments_body_only


    def detect_cell_text(self, cell_img, pad=30, scale_factor=3):
        # Convert to RGB and add padding
        img = cell_img.convert("RGB")
        w, h = img.size

        padded_img = PILImage.new("RGB", (w + 2*pad, h + 2*pad), (255, 255, 255))
        padded_img.paste(img, (pad, pad))

        # Upscale (important!)
        new_w, new_h = padded_img.size[0] * scale_factor, padded_img.size[1] * scale_factor
        padded_img = padded_img.resize((new_w, new_h), PILImage.BICUBIC)

        # Run PaddleOCR
        img_array = np.array(padded_img)
        try:
            result = self.paddle_ocr.ocr(img_array, cls=True)
            texts = [text for line in result for box, (text, conf) in line if conf > 0.3]
            return " ".join(texts) if texts else ""
        except:
            return ""

    def extract_df(self, col_segments_body_only, headers):
        table_data = {}
        max_rows = 0

        for i, col_seg in enumerate(col_segments_body_only):
            cells = self.crop_rows_from_horizontal_lines(col_seg, width_req_ratio=0.9)
            header_text = headers[i]
            col_texts = []

            for cell_img in cells:
                text = self.detect_cell_text(cell_img)
                col_texts.append(text)

            table_data[header_text] = col_texts
            max_rows = max(max_rows, len(col_texts))

        # Normalize column lengths
        for col in table_data:
            while len(table_data[col]) < max_rows:
                table_data[col].append("")

        df = pd.DataFrame(table_data)
        
        return df

    

    def perspective_transform(self, image: PILImage.Image, box: np.ndarray) -> PILImage.Image:
        img_cv = np.array(image)

        # Convert and validate shape
        rect = np.array(box, dtype=np.float32)
        if rect.shape != (4, 2):
            raise ValueError(f"Expected 4 points with shape (4, 2), got shape {rect.shape} and data: {box}")

        # Compute width and height of new image
        widthA = np.linalg.norm(rect[2] - rect[3])
        widthB = np.linalg.norm(rect[1] - rect[0])
        maxWidth = int(max(widthA, widthB))

        heightA = np.linalg.norm(rect[1] - rect[2])
        heightB = np.linalg.norm(rect[0] - rect[3])
        maxHeight = int(max(heightA, heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype=np.float32)

        # Get transform and warp image
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img_cv, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)

        return PILImage.fromarray(warped)