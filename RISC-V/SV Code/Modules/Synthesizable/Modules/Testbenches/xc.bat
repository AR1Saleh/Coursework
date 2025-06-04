@echo off
call xv
if %ERRORLEVEL% GEQ 1 EXIT /B 1
vsim -c work.riscv_tb -do "run -all; quit" 
