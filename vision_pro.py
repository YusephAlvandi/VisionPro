"""
VisionPro v2 — Unified Face Detection & Object Counting
Author: Yuseph Alvandi
Description: AI-powered detection with YOLOv8m. Thick bounding boxes, centered labels.
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
from ultralytics import YOLO
import cv2
import os
from PIL import Image, ImageTk

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class VisionProApp:
    def __init__(self):
        self.window = ctk.CTk()
        self.window.title("VisionPro — Face & Object Detection")
        self.window.geometry("1100x800")
        self.window.configure(fg_color="#0a0a0a")
        
        self.image_path = None
        self.original_img = None
        self.current_img = None
        
        model_path = os.path.expanduser("~/python_projects/image_processing/models/yolov8m.pt")
        self.model = YOLO(model_path)
        self.confidence = ctk.DoubleVar(value=0.5)
        self.mode = ctk.StringVar(value="all")
        
        self.setup_ui()
    
    def setup_ui(self):
        header = ctk.CTkFrame(self.window, fg_color="transparent")
        header.pack(fill="x", pady=(20, 10), padx=30)
        ctk.CTkLabel(header, text="VisionPro", font=ctk.CTkFont(size=32, weight="bold"), text_color="#1E90FF").pack()
        ctk.CTkLabel(header, text="Face Detection & Object Counting — YOLOv8m", font=ctk.CTkFont(size=14), text_color="#AAAAAA").pack()
        
        toolbar = ctk.CTkFrame(self.window, fg_color="#1a1a1a", corner_radius=12, height=60)
        toolbar.pack(fill="x", padx=30, pady=(5, 10))
        toolbar.pack_propagate(False)
        ctk.CTkButton(toolbar, text="Open Image", command=self.open_image, width=130, height=40).pack(side="left", padx=(20, 5), pady=10)
        ctk.CTkButton(toolbar, text="Batch Folder", command=self.process_batch, width=130, height=40, fg_color="#E67E22").pack(side="left", padx=5, pady=10)
        ctk.CTkButton(toolbar, text="Save Result", command=self.save_result, width=130, height=40, fg_color="#2ECC71").pack(side="left", padx=5, pady=10)
        self.file_label = ctk.CTkLabel(toolbar, text="No image selected", text_color="#888888", font=ctk.CTkFont(size=12))
        self.file_label.pack(side="left", padx=15, pady=10)
        
        content = ctk.CTkFrame(self.window, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=30, pady=10)
        
        left = ctk.CTkFrame(content, fg_color="#1a1a1a", corner_radius=12, width=380)
        left.pack(side="left", fill="y", padx=(0, 10))
        left.pack_propagate(False)
        
        ctk.CTkLabel(left, text="Detection Mode", font=ctk.CTkFont(size=18, weight="bold"), text_color="#1E90FF").pack(pady=(20, 10))
        
        ctk.CTkRadioButton(left, text="All Objects (80 classes)", variable=self.mode, value="all", command=self.on_mode_change).pack(anchor="w", padx=30, pady=5)
        ctk.CTkRadioButton(left, text="Faces Only", variable=self.mode, value="faces", command=self.on_mode_change).pack(anchor="w", padx=30, pady=5)
        
        ctk.CTkLabel(left, text="Confidence Threshold", font=ctk.CTkFont(size=14, weight="bold"), text_color="#CCCCCC").pack(pady=(20, 5))
        ctk.CTkLabel(left, text="Lower = more detections", font=ctk.CTkFont(size=10), text_color="#888888").pack()
        ctk.CTkSlider(left, from_=0.1, to=0.9, variable=self.confidence, width=250, command=self.on_slider_change).pack()
        self.conf_label = ctk.CTkLabel(left, text="0.5", font=ctk.CTkFont(size=12), text_color="#1E90FF")
        self.conf_label.pack()
        
        self.count_label = ctk.CTkLabel(left, text="Detected: 0", font=ctk.CTkFont(size=22, weight="bold"), text_color="#4CAF50")
        self.count_label.pack(pady=(20, 5))
        
        self.objects_text = ctk.CTkTextbox(left, height=200, font=ctk.CTkFont(size=11))
        self.objects_text.pack(pady=10, padx=15, fill="x")
        
        self.status_label = ctk.CTkLabel(left, text="Ready — YOLOv8m loaded", text_color="#4CAF50", font=ctk.CTkFont(size=10))
        self.status_label.pack(pady=10)
        
        right = ctk.CTkFrame(content, fg_color="#1a1a1a", corner_radius=12)
        right.pack(side="right", fill="both", expand=True)
        self.preview_label = ctk.CTkLabel(right, text="No Image Loaded", font=ctk.CTkFont(size=16), text_color="#555555")
        self.preview_label.pack(expand=True)
    
    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not path: return
        self.image_path = path
        self.original_img = cv2.imread(path)
        self.file_label.configure(text=os.path.basename(path))
        self.detect_and_show()
    
    def on_mode_change(self):
        if self.original_img is not None: self.detect_and_show()
    
    def on_slider_change(self, value=None):
        self.conf_label.configure(text=f"{self.confidence.get():.1f}")
        if self.original_img is not None: self.detect_and_show()
    
    def detect_and_show(self):
        if self.original_img is None: return
        img = self.original_img.copy()
        results = self.model(img, conf=self.confidence.get(), verbose=False)
        
        object_count = 0
        self.objects_text.delete("1.0", "end")
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id]
                    
                    if self.mode.get() == "faces" and cls_name != "person":
                        continue
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    object_count += 1
                    
                    # Thick bounding box (4px)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
                    
                    # Centered label with number
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    label = f"#{object_count} {cls_name}"
                    
                    # Background rectangle for text
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(img, (cx - tw//2 - 5, cy - th//2 - 5), (cx + tw//2 + 5, cy + th//2 + 5), (0, 0, 0), -1)
                    cv2.putText(img, label, (cx - tw//2, cy + th//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    self.objects_text.insert("end", f"{object_count}. {cls_name} ({conf:.2f})\n")
        
        self.current_img = img
        self.count_label.configure(text=f"Detected: {object_count}")
        
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        max_w, max_h = 600, 500
        ratio = min(max_w / pil_img.width, max_h / pil_img.height)
        pw, ph = int(pil_img.width * ratio), int(pil_img.height * ratio)
        pil_img = pil_img.resize((pw, ph))
        tk_img = ImageTk.PhotoImage(pil_img)
        self.preview_label.configure(image=tk_img, text="")
        self.preview_label.image = tk_img
    
    def process_batch(self):
        input_dir = filedialog.askdirectory(title="Select Input Folder")
        if not input_dir: return
        output_dir = filedialog.askdirectory(title="Select Output Folder")
        if not output_dir: return
        
        supported = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        processed = 0
        total_objects = 0
        
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(supported):
                img = cv2.imread(os.path.join(input_dir, filename))
                if img is None: continue
                results = self.model(img, conf=self.confidence.get(), verbose=False)
                
                obj_count = 0
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            cls_id = int(box.cls[0])
                            cls_name = self.model.names[cls_id]
                            if self.mode.get() == "faces" and cls_name != "person": continue
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            obj_count += 1
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
                            cx, cy = (x1 + x2)//2, (y1 + y2)//2
                            label = f"#{obj_count} {cls_name}"
                            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(img, (cx - tw//2 - 5, cy - th//2 - 5), (cx + tw//2 + 5, cy + th//2 + 5), (0, 0, 0), -1)
                            cv2.putText(img, label, (cx - tw//2, cy + th//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                total_objects += obj_count
                processed += 1
                cv2.imwrite(os.path.join(output_dir, f"detected_{filename}"), img)
                self.status_label.configure(text=f"Processed: {filename} ({obj_count} detections)", text_color="#FFAA33")
                self.window.update()
        
        self.status_label.configure(text=f"Done! {processed} images.", text_color="#4CAF50")
        messagebox.showinfo("Complete", f"Processed {processed} images.\nTotal detections: {total_objects}")
    
    def save_result(self):
        if self.current_img is None: messagebox.showerror("Error", "No processed image!"); return
        path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        if path: cv2.imwrite(path, self.current_img)
    
    def run(self): self.window.mainloop()

if __name__ == "__main__":
    app = VisionProApp()
    app.run()
