# Matrix-inversion-for-image-processing

README.md
Image Affine Transformation Parallelization
This repository contains a Python script for performing affine transformations on images using OpenCV. The script demonstrates both serial and parallel execution for comparison.

Overview
The script utilizes OpenCV for image processing, Matplotlib for visualization, and multiprocessing for parallel execution. Affine transformations are applied to uploaded images, showcasing the difference in execution time between serial and parallel approaches.

Prerequisites
Make sure you have the following dependencies installed:

Python 3.x
OpenCV
Matplotlib
You can install the required packages using the following:

bash
Copy code
pip install opencv-python matplotlib
Usage
Clone the repository:
bash
Copy code
git clone https://github.com/your-username/affine-transformation-parallelization.git
cd affine-transformation-parallelization
Upload images:

Execute the script in a Jupyter notebook or Google Colab.
Upload multiple images when prompted.
Run the script:

The script applies affine transformations to the uploaded images both serially and in parallel.
Execution times for both approaches will be displayed.
Script Explanation
parallel_processing: Function for parallel execution of affine transformations using multiprocessing.
get_image_size: Helper function to retrieve the dimensions of an image.
Serial execution: Affine transformations are applied sequentially.
Parallel execution: Affine transformations are parallelized using multiprocessing.
Example
python
Copy code
import cv2
from matplotlib import pyplot as plt
import time
from google.colab.patches import cv2_imshow
from multiprocessing import Process, Manager
from google.colab import files

# ... (Rest of the script)

# Display images for serial execution
plt.subplot(121)
plt.imshow(img)
plt.title('Input (Serial)')

plt.subplot(122)
plt.imshow(dst_serial)
plt.title('Output (Serial)')

plt.show()

# Display images for parallel execution
plt.subplot(121)
plt.imshow(img)
plt.title('Input (Parallel)')

plt.subplot(122)
plt.imshow(dst_parallel)
plt.title('Output (Parallel)')

plt.show()
Contributions
Contributions are welcome! Feel free to open issues or pull requests.

License
This project is licensed under the MIT License.

Acknowledgments
Thanks to the OpenCV and Matplotlib communities for their valuable contributions.
Note: Adjust the clone URL (https://github.com/your-username/affine-transformation-parallelization.git) based on your GitHub username and repository name
