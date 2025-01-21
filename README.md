# Image Clustering DEXA
This repository contains code for processing DEXA images, extracting embeddings, clustering them, and visualizing the results.

## Requirements
Make sure you have the required packages installed. You can install them using the following command:

```
pip3 install -r requirements.txt
```
## Running the Project
### 1. Setup and Run Data Conversion
To set up the environment and run the data conversion scripts, use the setup_and_run_data_convert.sh script:
```
bash setup_and_run_data_convert.sh
```

This script will:

1. Install the required packages.
2. Download the dataset.
3. Run the data conversion scripts.
4. Upload the converted data to the specified location.

### 2. Running the Main Script
After converting the data, you can run the main script to perform clustering and visualization:
```
python3 main_ukb.py --dataset_path ./converted_data --output ./output --num_clusters 4 --model xrv_vae --use_cache
```
### 3. Running the Background Removal Test
To test the background removal functionality, use the main_remove_bk_test.py script:
```
python3 main_remove_bk_test.py --dataset_path ./converted_data --output ./output_removebk
```

### 4. Running the 3D t-SNE Visualization
To run the 3D t-SNE visualization, use the main_3d_tsne.py script:
```
python3 main_3d_tsne.py --dataset_path ./output_4clusters_5000images --output ./output_3d_tsne
```

## Additional Scripts
Data Conversion Script
The dataset_conversion.py script processes the DEXA data and creates a tarball:
```
python3 dataset_conversion.py --dataset_path ./data --output ./output_data --tarball_name output --remove_zip --remove_unused_dcm
```

### Clustering Script
The `cluster.py` script contains functions for KMeans clustering and t-SNE embeddings.

### Utilities
The `utils.py` script contains various utility functions used throughout the project.

### Transforms
The `transforms.py` script contains custom image transformation functions.

### Embeddings
The `embeddings.py` script contains functions to extract embeddings from the dataset.

### Datasets
The `datasets.py` script contains dataset classes for loading and processing the DEXA images.

### License
This project is licensed under the BSD 3-Clause License. See the LICENSE file for details.