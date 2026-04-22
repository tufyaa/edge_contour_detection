# Minimal makefile for Sphinx documentation

SPHINXBUILD ?= sphinx-build
SOURCEDIR = .
BUILDDIR = _build

.PHONY: clean html

clean:
	rm -rf $(BUILDDIR)
	rm -rf autoapi

html:
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html
