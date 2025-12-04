@echo off
if "%1"=="" (
    echo Usage: texpdf filename.tex
    exit /b
)
pdflatex %1
pdflatex %1
del "%~n1.aux" "%~n1.log" "%~n1.out" 2>nul
echo Done! PDF created: %~n1.pdf
