
for file in ./Config/cutters/*; do
    echo $file
    python3 ./Program/MCP.py --input $file

done
