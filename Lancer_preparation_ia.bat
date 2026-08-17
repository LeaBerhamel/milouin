@echo off
setlocal
title Suivi Canards - Preparation des donnees IA
cd /d "%~dp0"

REM Lance l'interface de PREPARATION DES DONNEES (tri couleur/espece,
REM entrainement, gestion des versions) -- pas a confondre avec Lancer.bat,
REM qui lance l'interface d'ANALYSE des photos.

echo ============================================================
echo   Suivi Canards - Preparation des donnees IA (couleur / espece)
echo ============================================================
echo.

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERREUR] Python n'est pas installe, ou n'est pas accessible
    echo          depuis cette fenetre.
    echo.
    echo Installe Python depuis https://www.python.org/downloads/
    echo IMPORTANT : lors de l'installation, coche la case
    echo "Add python.exe to PATH" avant de cliquer sur Install.
    echo.
    pause
    exit /b 1
)

python -c "import cv2, pandas, ultralytics, torch, PIL, matplotlib, openpyxl, sklearn" >nul 2>nul
if %errorlevel% neq 0 (
    echo ============================================================
    echo   Premiere utilisation detectee
    echo ============================================================
    echo.
    echo Installation des dependances necessaires ^(une seule fois^).
    echo Certaines ^(PyTorch notamment^) sont volumineuses : selon ta
    echo connexion, ca peut prendre de 2 a 15 minutes.
    echo.
    echo    NE FERME PAS CETTE FENETRE, meme si rien ne semble
    echo    bouger pendant quelques dizaines de secondes : c'est
    echo    normal pendant que pip prepare le telechargement.
    echo.
    echo Progression du telechargement affichee ci-dessous :
    echo ------------------------------------------------------------
    python -m pip install --disable-pip-version-check -r requirements.txt
    if %errorlevel% neq 0 (
        echo.
        echo [ERREUR] L'installation des dependances a echoue.
        echo Copie le message ci-dessus si tu demandes de l'aide.
        pause
        exit /b 1
    )
    echo ------------------------------------------------------------
    echo Installation terminee.
    echo.
)

python scripts\interface_preparation_ia.py
if %errorlevel% neq 0 (
    echo.
    echo [ERREUR] L'interface s'est fermee de maniere inattendue.
    pause
)