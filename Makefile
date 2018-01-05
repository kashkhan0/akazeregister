CC = g++

all:  akazeregister4   akklines 

akazeregister4: akazeregister4.cpp   
	g++  -std=c++11  akazeregister4.cpp -o akazeregister4 `pkg-config opencv --cflags --libs`  

akklines: akklines.cpp akkutil.hpp json.hpp
	g++  -std=c++11  akklines.cpp -o akklines `pkg-config opencv --cflags --libs`  

thumbexp: thumbexp.cpp    oc.hpp
	g++  -std=c++11  thumbexp.cpp -o thumbexp `pkg-config opencv --cflags --libs`  `pkg-config libcurl --cflags --libs`   
applyclahe: applyclahe.cpp    oc.hpp
	g++  -std=c++11  applyclahe.cpp -o applyclahe `pkg-config opencv --cflags --libs`  `pkg-config libcurl --cflags --libs`   
applyclahehls: applyclahehls.cpp    
	g++  -std=c++11  applyclahehls.cpp -o applyclahehls `pkg-config opencv --cflags --libs`  `pkg-config libcurl --cflags --libs`   
 
gcutjs: gcutjs.cpp   oc.hpp
	g++  -std=c++11  gcutjs.cpp -o gcutjs `pkg-config opencv --cflags --libs`  `pkg-config libcurl --cflags --libs`   

akk4: akk4.cpp
	g++  -std=c++11  akk4.cpp -o akk4 `pkg-config opencv --cflags --libs`  
		
findhomog: findhomog.cpp 
	g++  -std=c++11  findhomog.cpp -o findhomog `pkg-config opencv --cflags --libs`  

findshift: findshift.cpp 
	g++  -std=c++11  findshift.cpp -o findshift `pkg-config opencv --cflags --libs`  
	
akshift: akshift.cpp findhomoglib.cpp
	g++  -std=c++11  akshift.cpp -o akshift `pkg-config opencv --cflags --libs`  

clahec: clahec.cpp
	g++  -std=c++11  clahec.cpp -o clahec `pkg-config opencv --cflags --libs`  
nld: nld.cpp
	g++  -std=c++11  nld.cpp -o nld `pkg-config opencv --cflags --libs`  

lsd_lines: lsd_lines.cpp
	g++  -std=c++11  lsd_lines.cpp -o lsd_lines `pkg-config opencv --cflags --libs`  

exp4: exp4.cpp
	g++  -std=c++11  exp4.cpp -o exp4 `pkg-config opencv --cflags --libs`  

equalize1:equalize1.cpp
	g++ equalize1.cpp -o equalize1 `pkg-config opencv --cflags --libs`  


keypoints:keypoints.cpp
	g++ keypoints.cpp -o keypoints `pkg-config opencv --cflags --libs`  
akfeat:akfeat.cpp
	g++ akfeat.cpp -o akfeat `pkg-config opencv --cflags --libs`  

hist: hist.cpp
	g++ hist.cpp -o hist `pkg-config opencv --cflags --libs`  

dem: dem.cpp
	g++ dem.cpp -o dem `pkg-config opencv --cflags --libs` 





clean:
	rm -rf *o  akklines akazeregister4
