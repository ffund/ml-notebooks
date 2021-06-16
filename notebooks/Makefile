# All markdown files are considered sources
#SOURCES := $(wildcard *.md)
SOURCES := 1-python-numpy-tutorial.md
SOURCES += 1-colab-tour.md
SOURCES += 1-print-colab.md
SOURCES += 1-exploratory-data-analysis.md
SOURCES += 1-data-detective.md
SOURCES += 1-explore-hw.md
SOURCES += 2-compute-by-hand.md
SOURCES += 2-regression-r2.md

# Convert the list of source files (Markdown files )
# into a list of output files
NBS := $(patsubst %.md,%.ipynb,$(SOURCES))

# The ones to also execute and generate PDFs
EXECS := 1-python-numpy-tutorial.md
EXECS += 1-exploratory-data-analysis.md
EXECS += 1-data-detective.md
EXECS += 2-compute-by-hand.md
EXECS += 2-regression-r2.md

NBSEXEC := $(patsubst %.md,%.nbconvert.ipynb,$(EXECS))
PDFS := $(patsubst %.md,%.pdf,$(EXECS))

PANDOCFLAGS=--pdf-engine=xelatex\
         -V mainfont='Fira Sans' \
         -V geometry:margin=1in \
         --highlight-style pygments \
	 --listings --variable urlcolor=Maroon \
	 -H style/listings-setup.tex -H style/keystroke-setup.tex -H style/includes.tex

%.ipynb: %.md
	pandoc  --self-contained  $^ -o $@

%.nbconvert.ipynb: %.ipynb
	jupyter nbconvert --to notebook --execute --ExecutePreprocessor.allow_errors=True --ExecutePreprocessor.timeout=360 $^

%.pdf: %.nbconvert.ipynb
	pandoc $^ $(PANDOCFLAGS) -o $@

all: $(NBS)

notebooks: $(NBS) 

executed: $(NBSEXEC)

pdfs: $(PDFS)

clean: 
	rm -f *.ipynb

