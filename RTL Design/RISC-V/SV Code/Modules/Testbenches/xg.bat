@echo off
call xv.bat
if %ERRORLEVEL% GEQ 1 EXIT /B 1
vsim -gui work.riscv_tb -do "do wave.do;run -all; quit" 

