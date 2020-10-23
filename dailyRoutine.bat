timeout 10
call activate research
cd C:\Users\xudong\Documents\github\coresearch\鹏晖量化1号
timeout 5
jupyter nbconvert --to python research.ipynb
python research.py
jupyter nbconvert --to python analysis.ipynb
python analysis.py
jupyter nbconvert --to python product.ipynb
start cmd /c "python product.py"

cd C:\Users\xudong\Documents\github\coresearch\培宏量化1号
jupyter nbconvert --to python research.ipynb
python research.py
jupyter nbconvert --to python analysis.ipynb
python analysis.py
jupyter nbconvert --to python product.ipynb
start cmd /c "python product.py"

date /t && time /t
timeout 36000
