mport cv2
import numpy as np
from matplotlib import pyplot as plt
import time
from google.colab.patches import cv2_imshow  # For displaying images in Colab
from multiprocessing import Process, Manager
from google.colab import files  # For file upload in Colab

def get_image_size(img):
    height, width, channels = img.shape
    return width, height, channels

def parallel_processing(img, pts1, pts2, cols, rows, result_dict):
    M = cv2.getAffineTransform(pts1, pts2)
    result_dict['dst_parallel'] = cv2.warpAffine(img, M, (cols, rows))

# Upload multiple images
uploaded = files.upload()

# Process each uploaded file
for uploaded_file_name in uploaded.keys():
    print(f"Uploaded file: {uploaded_file_name}")

    # Read the uploaded image
    img = cv2.imdecode(np.frombuffer(uploaded[uploaded_file_name], np.uint8), cv2.IMREAD_COLOR)
    width, height, channels = get_image_size(img)
    print(f"Image Size: Width={width}, Height={height}, Channels={channels}")

    # Define the transformation points
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

    # Serial execution
    start_time_serial = time.time()
    M = cv2.getAffineTransform(pts1, pts2)
    dst_serial = cv2.warpAffine(img, M, (width, height))
    end_time_serial = time.time()
    print(f"Serial execution time: {end_time_serial - start_time_serial} seconds")

    # Display the images for serial execution
    plt.subplot(121)
    plt.imshow(img)
    plt.title('Input (Serial)')

    plt.subplot(122)
    plt.imshow(dst_serial)
    plt.title('Output (Serial)')

    plt.show()

    # Parallel execution
    manager = Manager()
    result_dict = manager.dict()

    start_time_parallel = time.time()

    # Create a process for parallel execution
    process_parallel = Process(target=parallel_processing, args=(img, pts1, pts2, width, height, result_dict))
    process_parallel.start()
    process_parallel.join()  # Wait for the process to finish

    dst_parallel = result_dict['dst_parallel']

    end_time_parallel = time.time()
    print(f"Parallel execution time: {end_time_parallel - start_time_parallel} seconds")

    # Display the images for parallel execution
    plt.subplot(121)
    plt.imshow(img)
    plt.title('Input (Parallel)')

    plt.subplot(122)
    plt.imshow(dst_parallel)
    plt.title('Output (Parallel)')

    plt.show()
