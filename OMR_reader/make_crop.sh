
for file in ./ProInput/*; do
    echo $file
    python ./Program/crop_pro.py --input $file

done

