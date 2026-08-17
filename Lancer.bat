@echo off
setlocal
title Suivi Canards
cd /d "%~dp0"

REM Astuce : tu peux glisser-deposer un dossier de photos (et/ou un
REM fichier .xlsx/.csv d'observations) directement sur l'icone Lancer.bat.
REM Windows transmet alors ces elements en argument (%*) et l'interface
REM les pre-remplit automatiquement. Un simple double-clic marche aussi,
REM sans rien deposer.

echo ============================================================
echo   Suivi Canards - Lancement de l'interface
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

python scripts\interface_graphique.py %*
if %errorlevel% neq 0 (
    echo.
    echo [ERREUR] L'interface s'est fermee de maniere inattendue.
    pause
)