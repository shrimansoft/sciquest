
for file in ../sciquest/*/*; do
    echo $file
    python ./Program/crop_pro.py --input $file 

done

