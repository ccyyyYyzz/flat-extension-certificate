PYTHON ?= python
LATEXMK ?= latexmk

.PHONY: all figures artifacts tests letter supplement reproduce clean

all: letter supplement

figures:
	cd ../.. && $(PYTHON) paper/prl/figures/make_c7_flat_extension.py

artifacts:
	cd ../.. && $(PYTHON) examples/run_spacetime_witness.py
	cd ../.. && $(PYTHON) examples/run_witness_power.py

tests:
	cd ../.. && $(PYTHON) -m unittest discover -s tests -v

letter: figures
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex

supplement:
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error supplement.tex

reproduce: tests artifacts figures all

clean:
	$(LATEXMK) -C main.tex
	$(LATEXMK) -C supplement.tex
