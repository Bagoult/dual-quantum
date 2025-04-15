# Variables
COMPILER = pdflatex
PAPER_DIR = Paper
MAIN_TEX = main.tex
OUTPUT_PDF = main.pdf

# Target to build the PDF
all:
	cd $(PAPER_DIR) && $(COMPILER) $(MAIN_TEX)

# Clean auxiliary files
clean:
	cd $(PAPER_DIR) && rm -f *.aux *.log *.out *.toc *.lof *.lot *.bbl *.blg *.synctex.gz

# Clean everything including PDF
distclean: clean
	cd $(PAPER_DIR) && rm -f $(OUTPUT_PDF)

.PHONY: all clean distclean