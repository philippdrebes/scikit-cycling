.PHONY: all clean test
PYTHON=python
NOSETESTS=nosetests

all:
	$(PYTHON) setup.py build_ext --inplace

clean:
	find . -name "*.so" -o -name "*.pyc" -o -name "*.md5" -o -name "*.pyd" | xargs rm -f
	find . -name "*.pyx" -exec ./tools/rm_pyx_c_file.sh {} \;
	rm -rf coverage
	rm -rf dist
	rm -rf build

test:
	$(NOSETESTS) -s -v skcycling

# doctest:
# 	$(PYTHON) -c "import skcycling, sys, io; sys.exit(skcycling.doctest_verbose())"

coverage:
	$(NOSETESTS) skcycling -s -v --with-coverage --cover-package=skcycling

html:
	conda install -y sphinx sphinx_rtd_theme
	export SPHINXOPTS=-W; make -C doc html
