timeout 10
cd d:\pkl
del *.* /F /Q
cd c:\Program Files (x86)\cwRsync\bin
rsync -rlptD --progress 115.206.232.22::ftp /cygdrive/d/pkl
call activate research

cd C:\Users\xudong\Documents\github\coresearch\�������1��
timeout 5
jupyter nbconvert --to python research.ipynb
python research.py
jupyter nbconvert --to python analysis.ipynb
python analysis.py
jupyter nbconvert --to python product.ipynb
start cmd /c "python product.py"

cd C:\Users\xudong\Documents\github\coresearch\�������2��
timeout 5
jupyter nbconvert --to python research.ipynb
python research.py
jupyter nbconvert --to python analysis.ipynb
python analysis.py
jupyter nbconvert --to python product.ipynb
start cmd /c "python product.py"

cd C:\Users\xudong\Documents\github\coresearch\��������1��
timeout 5
jupyter nbconvert --to python research.ipynb
python research.py
jupyter nbconvert --to python analysis.ipynb
python analysis.py
jupyter nbconvert --to python product.ipynb
start cmd /c "python product.py"

cd C:\Users\xudong\Documents\github\coresearch\�������1��
jupyter nbconvert --to python research.ipynb
python research.py
jupyter nbconvert --to python analysis.ipynb
python analysis.py
jupyter nbconvert --to python product.ipynb
start cmd /c "python product.py"

date /t && time /t
timeout 36000
