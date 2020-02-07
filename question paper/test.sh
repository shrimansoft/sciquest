i=1
for file in ./class_8/*
do
	echo $file
	mv  "$file"  "./class_8/Question_$((i++)).png"
done
