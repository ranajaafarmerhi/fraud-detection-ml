# Credit Card Fraud Detection
This project uses machine learning to detect credit card fraud. 
It compares Logistic Regression and Random Forest models, performs threshold tuning, 
and evaluates performance using classification reports, ROC-AUC, and cross-validation.
## Requirements
- Python 3.10+
- pandas
- numpy
- scikit-learn
## How to run
1. Clone the repository:
   git clone https://github.com/USERNAME/fraud-detection-ml.git
2. Navigate to the project folder:
   cd fraud-detection-ml
3. Create and activate a virtual environment:
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
4. Install required packages:
   pip install -r requirements.txt
5. Download `creditcard.csv` and place it in the `data/` folder.
6. Run preprocessing and training scripts:
   python src/preprocess.py
   python src/train.py
## Note
The dataset is large (~143 MB) and is not included in the repository. 
You need to download it locally from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) 
and place it in the `data/` folder.
## Results
- Logistic Regression ROC-AUC: ~0.979
- Random Forest ROC-AUC: ~0.951
- Threshold tuning improves precision for fraud detection.
- Cross-validation used to validate models' stability.