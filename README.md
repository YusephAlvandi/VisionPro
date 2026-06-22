VisionPro

This is an AI-powered face detection and object counting (all-in-one) tool. YOLOv8m (a deep neural network trained on 80 object classes) is utilized for developing this tool. Works offline.

Built with Python, OpenCV, and CustomTkinter.


WHAT IT DOES

VisionPro detects, labels, and counts objects and persons in images. Two modes available:

- Faces Only; detects and counts only people
- All Objects; detects all 80 classes (person, car, dog, cat, bottle, laptop, and more)

A confidence slider controls detection sensitivity. Thick bounding boxes with centered labels make results clearly visible. Batch folder processing supported.


DEPENDENCIES

Python libraries (install once):
pip install ultralytics opencv-python pillow customtkinter numpy

Model file (one-time download, place in ~/models/):
yolov8m.pt (48 MB); downloaded automatically on first run


HOW TO RUN

python vision_pro.py


AUTHOR

Yuseph Alvandi
PhD in Optics and Laser Physics
Python Developer and Image Processing Specialist

GitHub: https://github.com/YusephAlvandi


LICENSE

MIT License
