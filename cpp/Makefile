MRPT_PATH=../../mrpt
EIGEN_PATH=../eigen

CPPFLAGS=-I$(MRPT_PATH) -I$(EIGEN_PATH)
CXXFLAGS=-O3 -std=c++11 -fexceptions
LDLIBS=-lgomp

test : $(MRPT_PATH)/cpp/Mrpt.h

.PHONY: clean
clean:
	$(RM) test *.o
