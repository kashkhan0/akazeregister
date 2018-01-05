F1=$1
F2=$(echo $F1 | cut -d'.' -f 1)
#echo $F2
#wc $FILE1
#ls $FILE1
# g++ -w -std=c++14 $F1 -o $F2 `pkg-config opencv --cflags --libs` `pkg-config libcurl --cflags --libs` -I../misc -lsqlite3 -fopenmp -g
g++ -w -std=c++14 $F1 -o $F2 `pkg-config opencv --cflags --libs` `pkg-config libcurl --cflags --libs` `pkg-config Magick++ --cflags --libs`  -I../misc -lsqlite3 -g

