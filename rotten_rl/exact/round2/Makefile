# Compila el solver de la ronda de 2 cartas (ver README.md).
CXX ?= g++
CXXFLAGS ?= -O3 -march=native -std=c++17 -Wall -Wextra

build/round2: round2.cpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) -o $@ $<

.PHONY: clean
clean:
	rm -rf build
