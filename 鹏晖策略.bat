timeout 10
call activate research
cd C:\Users\xudong\Documents\github\coresearch\��������1��
timeout 5
jupyter nbconvert --to python research.ipynb
python research.py
jupyter nbconvert --to python analysis.ipynb
python analysis.py
jupyter nbconvert --to python product.ipynb
start cmd /c "python product.py"

