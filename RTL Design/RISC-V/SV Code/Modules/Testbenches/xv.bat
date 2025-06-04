@echo off
del wlf*
rmdir /s /q work
vlog -work work   -sv -stats=none riscv_tb.sv 
if %ERRORLEVEL% GEQ 1 EXIT /B 1
vlog -work work   -sv -stats=none ../ALU.sv 
if %ERRORLEVEL% GEQ 1 EXIT /B 1
vlog -work work   -sv -stats=none ../ALU_Ctrl.sv 
if %ERRORLEVEL% GEQ 1 EXIT /B 1
vlog -work work   -sv -stats=none ../Branch_Adder.sv 
if %ERRORLEVEL% GEQ 1 EXIT /B 1
vlog -work work   -sv -stats=none ../controller.sv 
if %ERRORLEVEL% GEQ 1 EXIT /B 1
vlog -work work   -sv -stats=none ../Data_memory.sv 
if %ERRORLEVEL% GEQ 1 EXIT /B 1
vlog -work work   -sv -stats=none ../imm_gen.sv 
if %ERRORLEVEL% GEQ 1 EXIT /B 1
vlog -work work   -sv -stats=none ../instruction_memory.sv 
if %ERRORLEVEL% GEQ 1 EXIT /B 1
vlog -work work   -sv -stats=none ../Mux_2x1.sv 
if %ERRORLEVEL% GEQ 1 EXIT /B 1
vlog -work work   -sv -stats=none ../PC.sv 
if %ERRORLEVEL% GEQ 1 EXIT /B 1
vlog -work work   -sv -stats=none ../PC_Adder.sv 
if %ERRORLEVEL% GEQ 1 EXIT /B 1
vlog -work work   -sv -stats=none ../Registers.sv 
if %ERRORLEVEL% GEQ 1 EXIT /B 1
vlog -work work   -sv -stats=none ../riscv.sv 
if %ERRORLEVEL% GEQ 1 EXIT /B 1