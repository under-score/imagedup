# imagedup
A proof of principle python script to find duplicated image regions in scientific papers

Developed on Sequoia 15.2

Created with the help of chatGPT 4o mini

Very early beta - still a lot of bugs

## background
https://towardsdatascience.com/bag-of-visual-words-in-a-nutshell-9ceea97ce0fb

https://www.nature.com/articles/d41586-021-03807-6

https://www.wjst.de/blog/sciencesurf/2021/09/a-comparison-of-image-duplication-software/

https://www.atsjournals.org/doi/10.1165/rcmb.2020-0419LE

https://retractionwatch.com/

https://blog.pubpeer.com/

## requirements
Neeeds python 3.8 eg *conda activate base38*

pip install *os sys cv2 sqlite3 numpy fitz sklearn tqdm faiss pytesseract datetime*

Needs tesseract *brew install tesseract*

Terminal *python imagedup.py <path_to_directory> <path_to_database>*

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

## benchmarks
10 min for 500 PDFs

TBC

## results
*./bovw*

*<path_to_directory>/images*

*<path_to_directory>/images_duplicated*
