#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Sund Dec 15 18:05:28 2024
Finds duplicated image regions
Runs on Mac under Sequoia 15.1
Needs Python 3.8
See readme
With help of ChatGPT o1

@author: wjst
"""

import os
import sys
import cv2
import sqlite3
import numpy as np
import fitz  # PyMuPDF for PDF processing
from sklearn.cluster import KMeans
from tqdm import tqdm
import faiss  # LSH-based library for fast similarity search
import pytesseract
from datetime import datetime


# Initialize start time
start_time = datetime.now()


def log(message):
    """Log messages with a timestamp indicating elapsed time."""
    elapsed_time = datetime.now() - start_time
    print(f"[{str(elapsed_time).split('.')[0]}] {message}")


def create_bovw_database(db_path):
    """Creates the SQLite database to store BoVW histograms."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bovw_histograms (
            pdf_name TEXT,
            page_number INTEGER,
            image_path TEXT UNIQUE,
            histogram BLOB
        )
    """)
    conn.commit()
    conn.close()


def save_bovw_to_database(db_path, pdf_name, page_number, image_path, histogram):
    """Saves a BoVW histogram to the SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO bovw_histograms (pdf_name, page_number, image_path, histogram)
            VALUES (?, ?, ?, ?)
        """, (pdf_name, page_number, image_path, histogram.tobytes()))
    except sqlite3.IntegrityError:
        log(f"Skipping duplicate entry for {image_path}")
    conn.commit()
    conn.close()


def extract_images_from_pdf(pdf_path, output_folder):
    """Extracts images from a PDF file and saves them in the specified folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    pdf_document = fitz.open(pdf_path)
    image_count = 0
    references = []

    log(f"Extracting images from {os.path.basename(pdf_path)}...")
    for page_number in range(len(pdf_document)):
        page = pdf_document[page_number]
        images = page.get_images(full=True)

        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_path = os.path.join(output_folder, f"{os.path.basename(pdf_path)}_page{page_number + 1}_img{img_index + 1}.{image_ext}")

            with open(image_path, "wb") as image_file:
                image_file.write(image_bytes)
            references.append((os.path.basename(pdf_path), page_number + 1, image_path))
            image_count += 1

    log(f"Extracted {image_count} images from {os.path.basename(pdf_path)}.")
    return references


def preprocess_image(image_path):
    """Preprocesses an image by masking text and straight lines."""
    image = cv2.imread(image_path)
    if image is None:
        return None

    # Mask text regions
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)
    try:
        text_boxes = pytesseract.image_to_boxes(gray)
        for box in text_boxes.splitlines():
            b = box.split()
            if len(b) < 5:
                continue
            x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
            cv2.rectangle(mask, (x, gray.shape[0] - y), (w, gray.shape[0] - h), 255, -1)
    except Exception as e:
        log(f"Error in text masking: {e}")
    image[mask == 255] = [0, 0, 0]

    # Mask straight lines
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 0), thickness=3)

    return image


def build_visual_vocabulary(image_paths, n_clusters=500):
    """Builds a visual vocabulary using KMeans clustering."""
    orb = cv2.ORB_create()
    descriptors_list = []

    log("Extracting descriptors from images...")
    for image_path in tqdm(image_paths, desc="Extracting descriptors"):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            log(f"Warning: Could not read image {image_path}. Skipping.")
            continue

        _, descriptors = orb.detectAndCompute(image, None)
        if descriptors is not None:
            descriptors_list.append(descriptors)

    if not descriptors_list:
        raise ValueError("No descriptors found in the dataset.")

    all_descriptors = np.vstack(descriptors_list)
    log(f"Total descriptors: {len(all_descriptors)}")

    n_clusters = min(n_clusters, len(all_descriptors))  # Ensure clusters <= descriptors
    log(f"Clustering {len(all_descriptors)} descriptors into {n_clusters} visual words...")

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    kmeans.fit(all_descriptors)

    return kmeans


def compute_bovw_histogram(kmeans, descriptors):
    """Computes a BoVW histogram for an image based on its descriptors."""
    n_clusters = len(kmeans.cluster_centers_)
    histogram = np.zeros(n_clusters, dtype=np.float32)  # Use float32 to match FAISS requirements
    for descriptor in descriptors:
        cluster_idx = kmeans.predict([descriptor])[0]
        histogram[cluster_idx] += 1
    return histogram


def build_and_store_bovw(image_references, db_path, kmeans):
    """Builds and stores BoVW histograms for all images."""
    orb = cv2.ORB_create()

    log("Building and storing BoVW histograms...")
    for pdf_name, page_number, image_path in tqdm(image_references):
        image = preprocess_image(image_path)
        if image is None:
            continue

        _, descriptors = orb.detectAndCompute(image, None)
        if descriptors is None:
            continue

        histogram = compute_bovw_histogram(kmeans, descriptors)
        save_bovw_to_database(db_path, pdf_name, page_number, image_path, histogram)


def build_lsh_index_from_bovw(db_path, nlist=100):
    """Builds an LSH index from the BoVW histograms stored in the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT histogram FROM bovw_histograms")
    rows = cursor.fetchall()

    histograms = [np.frombuffer(row[0], dtype=np.float32) for row in rows]
    histograms = np.array(histograms)

    if histograms.size == 0:
        raise ValueError("No histograms found in the database.")

    d = histograms.shape[1]  # Dimensionality of histograms
    quantizer = faiss.IndexFlatL2(d)
    lsh_index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

    log("Training LSH index...")
    lsh_index.train(histograms)
    lsh_index.add(histograms)

    conn.close()
    return lsh_index, histograms


def combine_overlapping_boxes(boxes):
    """Combines overlapping bounding boxes into single boxes."""
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda x: x[0])  # Sort by x1
    combined = [boxes[0]]

    for box in boxes[1:]:
        x1, y1, x2, y2 = box
        last_x1, last_y1, last_x2, last_y2 = combined[-1]

        # Check for overlap
        if x1 <= last_x2 and y1 <= last_y2:  # Overlapping
            combined[-1] = (
                min(last_x1, x1),
                min(last_y1, y1),
                max(last_x2, x2),
                max(last_y2, y2),
            )
        else:
            combined.append(box)

    return combined


def visualize_duplicates(image1_path, image2_path, boxes1, boxes2, color, output_folder, idx1, idx2):
    """Visualizes duplicate regions by placing images side by side and marking duplicates."""
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    if image1 is None or image2 is None:
        log(f"Error loading images: {image1_path} or {image2_path}")
        return

    combined_width = image1.shape[1] + image2.shape[1]
    combined_height = max(image1.shape[0], image2.shape[0])
    combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

    combined_image[: image1.shape[0], : image1.shape[1]] = image1
    combined_image[: image2.shape[0], image1.shape[1] :] = image2

    adjusted_boxes2 = [(x + image1.shape[1], y, x2 + image1.shape[1], y2) for x, y, x2, y2 in boxes2]

    for box1, box2 in zip(boxes1, adjusted_boxes2):
        cv2.rectangle(combined_image, (box1[0], box1[1]), (box1[2], box1[3]), color, 2)
        cv2.rectangle(combined_image, (box2[0], box2[1]), (box2[2], box2[3]), color, 2)

        midpoint1 = ((box1[0] + box1[2]) // 2, (box1[1] + box1[3]) // 2)
        midpoint2 = ((box2[0] + box2[2]) // 2, (box2[1] + box2[3]) // 2)
        cv2.line(combined_image, midpoint1, midpoint2, color, 2)

    output_path = os.path.join(output_folder, f"duplicate_{idx1}_{idx2}.jpg")
    cv2.imwrite(output_path, combined_image)
    log(f"Saved duplicate visualization: {output_path}")


def find_duplicates_with_lsh(image_references, histograms, lsh_index, duplication_dir, threshold=0.5):
    """
    Finds duplicate images using LSH, increasing specificity, and visualizes matches.
    """
    log("Searching for duplicates using LSH...")
    os.makedirs(duplication_dir, exist_ok=True)

    orb = cv2.ORB_create()

    color_palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    color_index = 0

    for idx1, histogram in enumerate(tqdm(histograms)):
        distances, indices = lsh_index.search(np.expand_dims(histogram, axis=0), k=10)

        for idx2, dist in zip(indices[0], distances[0]):
            if idx1 >= idx2 or dist >= threshold:
                continue  # Skip self-comparisons and low-confidence matches

            ref1 = image_references[idx1]
            ref2 = image_references[idx2]
            log(f"Duplicate found: {ref1[2]} <-> {ref2[2]}, Distance: {dist:.4f}")

            # Detect and match keypoints for visualization
            image1 = preprocess_image(ref1[2])
            image2 = preprocess_image(ref2[2])
            keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
            keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

            if descriptors1 is not None and descriptors2 is not None:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(descriptors1, descriptors2)
                matches = sorted(matches, key=lambda x: x.distance)[:20]  # Top 20 matches

                boxes1 = [(int(keypoints1[m.queryIdx].pt[0] - 10), int(keypoints1[m.queryIdx].pt[1] - 10),
                           int(keypoints1[m.queryIdx].pt[0] + 10), int(keypoints1[m.queryIdx].pt[1] + 10))
                          for m in matches]
                boxes2 = [(int(keypoints2[m.trainIdx].pt[0] - 10), int(keypoints2[m.trainIdx].pt[1] - 10),
                           int(keypoints2[m.trainIdx].pt[0] + 10), int(keypoints2[m.trainIdx].pt[1] + 10))
                          for m in matches]

                boxes1 = combine_overlapping_boxes(boxes1)
                boxes2 = combine_overlapping_boxes(boxes2)

                color = color_palette[color_index % len(color_palette)]
                color_index += 1

                visualize_duplicates(ref1[2], ref2[2], boxes1, boxes2, color, duplication_dir, idx1, idx2)


def process_directory_of_pdfs(directory, db_path, n_clusters=500):
    """Processes PDFs in a directory to extract images, build BoVW, and find duplicates."""
    create_bovw_database(db_path)

    all_image_references = []
    all_image_paths = []

    for pdf_file in os.listdir(directory):
        if pdf_file.lower().endswith(".pdf"):
            pdf_path = os.path.join(directory, pdf_file)
            image_folder = os.path.join(directory, "images", os.path.splitext(pdf_file)[0])

            image_references = extract_images_from_pdf(pdf_path, image_folder)
            all_image_references.extend(image_references)
            all_image_paths.extend([ref[2] for ref in image_references])

    kmeans = build_visual_vocabulary(all_image_paths, n_clusters=n_clusters)
    build_and_store_bovw(all_image_references, db_path, kmeans)

    histograms = []
    for ref in all_image_references:
        image_path = ref[2]
        image = preprocess_image(image_path)
        if image is None:
            continue

        _, descriptors = cv2.ORB_create().detectAndCompute(image, None)
        if descriptors is None:
            continue

        histogram = compute_bovw_histogram(kmeans, descriptors)
        histograms.append(histogram)

    histograms = np.array(histograms)
    lsh_index, _ = build_lsh_index_from_bovw(db_path)

    duplication_dir = os.path.join(directory, "images_duplication")
    find_duplicates_with_lsh(all_image_references, histograms, lsh_index, duplication_dir)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <path_to_directory> <path_to_database>")
        sys.exit(1)

    directory = sys.argv[1]
    db_path = sys.argv[2]

    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        sys.exit(1)

    process_directory_of_pdfs(directory, db_path)
