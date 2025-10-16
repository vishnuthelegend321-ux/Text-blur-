import cv2
import easyocr
import os
from PIL import Image

# --- Configuration ---
INPUT_FOLDER = 'input_images'
OUTPUT_FOLDER = 'output_images'
PDF_FILENAME = 'processed_document.pdf'

# This initializes the OCR reader. You can add more languages like ['en', 'hi']
reader = easyocr.Reader(['en']) 

# Create output folder if it doesn't exist
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Get a sorted list of image files to maintain order
try:
    image_files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
except FileNotFoundError:
    print(f"Error: Input folder '{INPUT_FOLDER}' not found.")
    exit()

# --- Main Processing Loop ---
for image_filename in image_files:
    print(f"Processing {image_filename}...")
    image_path = os.path.join(INPUT_FOLDER, image_filename)
    output_path = os.path.join(OUTPUT_FOLDER, image_filename)
    
    original_image = cv2.imread(image_path)
    image_with_boxes = original_image.copy()

    try:
        # Use easyocr to detect text
        results = reader.readtext(image_path)
        
        # If no text is found, save the original image and continue
        if not results:
            print(f"  -> No text detected. Saving original image.")
            cv2.imwrite(output_path, original_image)
            continue # Move to the next image

        # Loop through the detected text results
        for (bbox, text, prob) in results:
            # bbox is a list of 4 points: [top-left, top-right, bottom-right, bottom-left]
            (tl, tr, br, bl) = bbox
            # Cast points to integers
            tl = (int(tl[0]), int(tl[1]))
            br = (int(br[0]), int(br[1]))
            
            # Draw a green rectangle using the top-left and bottom-right points
            cv2.rectangle(image_with_boxes, tl, br, (0, 255, 0), 2)
        
        # Save the MODIFIED image
        cv2.imwrite(output_path, image_with_boxes)
        print(f"  -> Text found. Saved modified image to {output_path}")

    except Exception as e:
        print(f"  -> An error occurred: {e}. Saving original image.")
        cv2.imwrite(output_path, original_image)

# ... (The PDF generation part of your code remains the same) ...
