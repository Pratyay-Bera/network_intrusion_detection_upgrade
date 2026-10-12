conda create -n myenv python=3.10   
pip install -r requirements.txt  
python src/train_models.py 
conda activate myenv    
 streamlit run src/app.py     