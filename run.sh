git pull
rm log*
for ((i=$1; i<$2; i=i+1))
do
    touch "log_stats_proj_2_$i.txt"
    touch "log$i.txt"
    touch "out$i.txt"
    (sleep 1; python "trainer.py" $i "baseline" "geo-distributed" >"out$i.txt") &


done