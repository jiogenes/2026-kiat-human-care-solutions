
for i in `seq 1 30`;
do
    num=$(printf "%02d" $i)
    wget https://fenix.ur.edu.pl/~mkepski/ds/data/fall-${num}-cam0.mp4 -O /data/kiat/UR_fall_detection/fall-${num}-cam0.mp4
done

for i in `seq 1 40`;
do
    num=$(printf "%02d" $i)
    wget https://fenix.ur.edu.pl/~mkepski/ds/data/adl-${num}-cam0.mp4 -O /data/kiat/UR_fall_detection/adl-${num}-cam0.mp4
done