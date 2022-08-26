# For QGCN 
---------------------------------------------------------------------------------
##1. Prepare data
cp ../dataset/*_train.csv ./data_process/train/raw
cp ../dataset/*_test.csv ./data_process/test/raw
---------------------------------------------------------------------------------
##2. Train
vi train.py
change filename in line85 to the dataset you want to train
python train.py
---------------------------------------------------------------------------------
##3. Test
rm ./data_process/train/processed
rm ./data_process/test/processed
vi ./data_process/dataset.py
delete ".sample(frac=1)" in line33 and line44
vi predict.py
change filename in line7,42 and line9,33 to the dataset you want to test
change filename in line40 and line48 to name you want to save the results as
python predict.py