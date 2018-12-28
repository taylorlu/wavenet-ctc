
FFTW = fftw-3.3.4
PYTHON_VERSION = 3.6
PYTHON_INCLUDE = /usr/local/include/python$(PYTHON_VERSION)
NUMPY_INCLUDE := $(shell python$(PYTHON_VERSION) -c 'import numpy; print(numpy.get_include())')
BOOST_INCLUDE = /usr/local/include
BOOST_LIB = /usr/local/lib

export PYTHON_VERSION
export PYTHON_INCLUDE
export NUMPY_INCLUDE
export BOOST_INCLUDE
export BOOST_LIB

all: ops

.PHONY: ops

ops:
	$(MAKE) -C user_ops

clean:
	$(MAKE) -C user_ops clean
