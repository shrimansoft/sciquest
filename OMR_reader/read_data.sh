
for file in ./Croped/*; do
    echo $file
    python ./Program/read_pro.py --input $file &

done