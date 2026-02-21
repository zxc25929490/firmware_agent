@echo off
echo ==============================
echo 建立虛擬環境 .venv
echo ==============================

python -m venv .venv

echo.
echo ==============================
echo 啟用虛擬環境
echo ==============================

call .venv\Scripts\activate

echo.
echo ==============================
echo 升級 pip
echo ==============================

python -m pip install --upgrade pip

echo.
echo ==============================
echo 安裝 requirements.txt
echo ==============================

pip install -r requirements.txt

echo.
echo ==============================
echo 完成 ✅
echo ==============================

pause