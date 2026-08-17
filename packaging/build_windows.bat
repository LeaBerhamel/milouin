@echo off
REM Construit l'executable Windows (dist\SuiviCanards\SuiviCanards.exe).
REM A executer sur une machine Windows, depuis la racine du projet :
REM     packaging\build_windows.bat
REM
REM Pre-requis : Python + toutes les dependances de requirements.txt deja
REM installees (torch, ultralytics, opencv...), plus pyinstaller.

setlocal
cd /d "%~dp0\.."

echo ============================================================
echo   Suivi Canards - Construction de l'executable Windows
echo ============================================================
echo.

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERREUR] Python n'est pas installe ou pas dans le PATH.
    pause
    exit /b 1
)

python -c "import PyInstaller" >nul 2>nul
if %errorlevel% neq 0 (
    echo Installation de PyInstaller...
    python -m pip install pyinstaller
)

echo Construction en cours ^(peut prendre plusieurs minutes, torch/ultralytics sont lourds^)...
python -m PyInstaller packaging\SuiviCanards.spec --distpath dist --workpath build --noconfirm --collect-all module_corrections --hidden-import serveur_corrections

if %errorlevel% neq 0 (
    echo.
    echo [ERREUR] La construction a echoue. Copie le message ci-dessus si tu demandes de l'aide.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Termine. Executable dans dist\SuiviCanards\SuiviCanards.exe
echo   IMPORTANT : copie tout le dossier dist\SuiviCanards\ a la
echo   racine du projet ^(a cote de data\, models\, output\^) avant
echo   de le distribuer ou de le lancer.
echo ============================================================
pause