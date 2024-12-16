# imagedup
Finds duplicated image regions in scientific papers

Complete Script Structure

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
