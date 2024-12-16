# imagedup
Finds duplicated image regions in scientific papers

Developed on Sequoia 15.2


## requisites

Neeeds python 3.8 **conda activate base38**

pip install os sys cv2 sqlite3 numpy fitz sklearn tqdm faiss pytesseract datetime

Terminal **python imagedup.py <path_to_directory> <path_to_database>**

Created with the help of chatGPT 4o mini


## pipeline

Utility Functions:
log, create_bovw_database, save_bovw_to_database.


Image Extraction:
extract_images_from_pdf.


Preprocessing:
preprocess_image (handles masking of text and straight lines).


BoVW Functions:
build_visual_vocabulary.
compute_bovw_histogram.
build_and_store_bovw.


LSH Functions:
build_lsh_index_from_bovw.
find_duplicates_with_lsh.


Main Pipeline:
process_directory_of_pdfs.
