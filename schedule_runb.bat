Set COUNTER=3
:x


echo %Counter%
if "%Counter%"=="33" (
    echo "END!"
) else (
    
    echo "%Counter%"
    python create_schedule.py %Counter%
    set /A COUNTER+=1
    goto x
)