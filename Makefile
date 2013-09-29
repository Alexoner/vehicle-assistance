CXX=g++-4.7
LIBS=`pkg-config --libs opencv`
INC=`pkg-config --cflags opencv`
FLAGS=-g -Wall

lane:lane.o utils.o
	$(CXX) -o $@ $^ $(LIBS) $(FLAGS)
test:test.o
	$(CXX) -o $@ $^ $(LIBS) $(FLAGS)

lane.o:lane.cpp
	$(CXX) $^ -c -o $@ $(INC) $(FLAGS)
utils.o:utils.cpp
	$(CXX) $^ -c -o $@ $(INC) $(FLAGS)
test.o:test.cpp
	$(CXX) -c -o $@ $^ $(INC) $(FLAGS)
