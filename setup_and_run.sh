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

echo "Running the main script"
cd $cfolder
python3 main_ukb.py --dataset_path ~/Dataset/DEXA_1000images
# python3 main_remove_bk_test.py --dataset_path ~/Dataset/DEXA_1000images

# dx upload output* -r --path /WorkspaceData/.

