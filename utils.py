import cv2
import numpy as np
import os
import pandas as pd
import fitz
from skimage.metrics import structural_similarity as ssim
from PIL import Image

class ProfileMatcher:
    def __init__(self, template_root, excel_path, pdf_folder):
        self.template_root = template_root
        self.excel_path = excel_path
        self.pdf_folder = pdf_folder
        self.templates = {}
        self.reference_size = 300
        self.excel_data = None
        self.load_templates()
        self.load_excel_data()
        
    def load_templates(self):
        """Pre-load all template images"""
        for class_name in os.listdir(self.template_root):
            class_path = os.path.join(self.template_root, class_name)
            if os.path.isdir(class_path):
                class_images = []
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_path, img_file)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            standardized = self.scale_normalize(img)
                            class_images.append({
                                'original': img,
                                'standardized': standardized,
                                'filename': os.path.basename(img_path),
                                'class': class_name
                            })
                if class_images:
                    self.templates[class_name] = class_images
    
    def load_excel_data(self):
        """Load Excel data with relaxed name matching"""
        try:
            self.excel_data = pd.read_excel(self.excel_path)
        except Exception as e:
            print(f"Error loading Excel file: {e}")
            self.excel_data = None
    
    def find_matching_row(self, class_name):
        """Find matching row in Excel with flexible name matching"""
        if self.excel_data is None:
            return None
            
        clean_class = class_name.replace(" ", "").replace("-", "").replace("_", "").lower()
        for idx, row in self.excel_data.iterrows():
            if pd.notna(row['Class']):
                clean_excel = str(row['Class']).replace(" ", "").replace("-", "").replace("_", "").lower()
                if clean_excel == clean_class:
                    return row
        return None
    
    def get_pdf_as_image(self, class_name):
        """Convert first page of PDF to displayable image"""
        pdf_files = [f for f in os.listdir(self.pdf_folder) if f.lower().endswith('.pdf')]
        target_pdf = None
        
        clean_class = class_name.replace(" ", "").replace("-", "").replace("_", "").lower()
        for pdf_file in pdf_files:
            pdf_name = os.path.splitext(pdf_file)[0]
            clean_pdf = pdf_name.replace(" ", "").replace("-", "").replace("_", "").lower()
            if clean_pdf == clean_class:
                target_pdf = pdf_file
                break
        
        if target_pdf is None:
            return None
        
        pdf_path = os.path.join(self.pdf_folder, target_pdf)
        
        try:
            doc = fitz.open(pdf_path)
            page = doc.load_page(0)
            pix = page.get_pixmap()
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
            doc.close()
            return img
        except Exception as e:
            print(f"Error loading PDF: {e}")
            return None
    
    def scale_normalize(self, image):
        """Normalize image scale while preserving aspect ratio"""
        _, thresh = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return cv2.resize(image, (self.reference_size, self.reference_size))
        
        main_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main_contour)
        
        profile_region = image[y:y+h, x:x+w]
        scale_factor = min(self.reference_size/w, self.reference_size/h)
        new_width = int(w * scale_factor)
        new_height = int(h * scale_factor)
        resized = cv2.resize(profile_region, (new_width, new_height))
        
        pad_width = (self.reference_size - new_width) // 2
        pad_height = (self.reference_size - new_height) // 2
        padded = cv2.copyMakeBorder(resized, 
                                  pad_height, pad_height, 
                                  pad_width, pad_width, 
                                  cv2.BORDER_CONSTANT, value=255)
        
        return cv2.resize(padded, (self.reference_size, self.reference_size))
    
    def preprocess_user_image(self, image):
        """Prepare user image for comparison"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        return self.scale_normalize(gray)
    
    def find_similar_profiles(self, user_image, max_matches=5):
        """Find matching profiles with scale normalization"""
        processed_user = self.preprocess_user_image(user_image)
        
        matches = []
        for class_name, template_list in self.templates.items():
            for template in template_list:
                similarity = ssim(processed_user, template['standardized'], full=True)[0]
                matches.append({
                    'similarity': similarity,
                    'class': class_name,
                    'image': template['original'],
                    'processed': template['standardized'],
                    'filename': template['filename']
                })
        
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Get best matches (up to max_matches)
        results = []
        seen_classes = set()
        for match in matches:
            if match['class'] not in seen_classes:
                results.append(match)
                seen_classes.add(match['class'])
                if len(results) >= max_matches:
                    break
        
        return processed_user, results