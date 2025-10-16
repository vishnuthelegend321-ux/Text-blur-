import cv2
import pytesseract
from pytesseract import Output
import os
from PIL import Image

# --- Configuration ---
# Set the path to your Tesseract installation if needed
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

INPUT_FOLDER = 'input_images'
OUTPUT_FOLDER = 'output_images'
PDF_FILENAME = 'processed_document.pdf'

# Create output folder if it doesn't exist
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Get a sorted list of image files to maintain order
try:
    image_files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))])
except FileNotFoundError:
    print(f"Error: Input folder '{INPUT_FOLDER}' not found. Please create it and add your images.")
    exit()

processed_images_for_pdf = []

# --- Main Processing Loop ---
for image_filename in image_files:
    print(f"Processing {image_filename}...")
    image_path = os.path.join(INPUT_FOLDER, image_filename)
    output_path = os.path.join(OUTPUT_FOLDER, image_filename)
    
    # Read the original image
    original_image = cv2.imread(image_path)
    
    # Make a copy for drawing boxes on, to keep the original clean
    image_with_boxes = original_image.copy()

    try:
        # Use pytesseract to get detailed data about text
        d = pytesseract.image_to_data(image_with_boxes, output_type=Output.DICT)
        
        # Check if any text was detected. If not, this will raise an error later.
        num_boxes = len(d['text'])
        if num_boxes == 0 or all(text.isspace() for text in d['text']):
            # This line forces the 'except' block to run if no text is found
            raise ValueError("No text detected in image.")

        # Loop over all detected text instances
        for i in range(num_boxes):
            # Only process boxes with a confidence score and actual text
            if int(d['conf'][i]) > 60 and not d['text'][i].isspace() and d['text'][i] != '':
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                # Draw a green rectangle around the detected text
                cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Save the MODIFIED image (with boxes)
        cv2.imwrite(output_path, image_with_boxes)
        print(f"  -> Text found. Saved modified image to {output_path}")

    except Exception as e:
        # This block runs if the 'try' block fails (e.g., no text found)
        print(f"  -> No text detected or error processing: {e}. Saving original image.")
        # Save the ORIGINAL, unmodified image
        cv2.imwrite(output_path, original_image)

# --- PDF Generation (Download Step) ---
# Get a sorted list of the processed images from the output folder
final_image_files = sorted([f for f in os.listdir(OUTPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))])

if not final_image_files:
    print("No images found in the output folder to create a PDF.")
else:
    # Open the first image and create a list of the rest to append
    first_image_path = os.path.join(OUTPUT_FOLDER, final_image_files[0])
    first_image = Image.open(first_image_path).convert('RGB')

    append_images = []
    for img_file in final_image_files[1:]:
        img_path = os.path.join(OUTPUT_FOLDER, img_file)
        append_images.append(Image.open(img_path).convert('RGB'))

    # Save all images into a single PDF
    first_image.save(PDF_FILENAME, save_all=True, append_images=append_images)
    print(f"\nSuccessfully created PDF: {PDF_FILENAME}")
