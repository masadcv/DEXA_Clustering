cfolder=$(pwd)
echo "Installing the required packages"
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip3 install -r requirements.txt

echo "Downloading the dataset"
cd ~
mkdir Dataset
cd Dataset
mkdir DEXA_1000images
cd DEXA_1000images

cp /mnt/project/Bulk/Whole\ Body\ DXA/DXA/10 . -R

# put this in new window to check if the files are copied
# watch "ls ~/Dataset/DEXA_1000images/10 -l | wc -l" 

cd $cfolder
echo "Running the data conversion scripts"
python3 data_conversion.py --dataset_path ~/Dataset/DEXA_1000images --output_path ./converted_data

dx upload ./converted_data/output.tar.gz /WorkspaceData/.

echo "Running the main script"

python3 main_ukb.py --dataset_path ./converted_data
# python3 main_remove_bk_test.py --dataset_path ~/Dataset/DEXA_1000images

dx upload output* -r --path /WorkspaceData/.
dx terminate $DX_JOB_ID

