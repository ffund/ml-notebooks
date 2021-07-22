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
SOURCES += 2-linear-regression-deep-dive.md
SOURCES += 2-linear-regression-case-study.md
SOURCES += 3-gradient-descent-deep-dive.md
SOURCES += 2-advertising-hw.md
SOURCES += 3-bias-variance-model-selection-deep-dive.md
SOURCES += 3-regularization-deep-dive.md
SOURCES += 3-neural-model-selection-hw.md
SOURCES += 4-logistic-regression-in-depth.md
SOURCES += 4-logistic-regression-digits.md
SOURCES += 4-compas-case-study.md
SOURCES += 4-hw-logistic-regression.md
SOURCES += 5-k-nearest-neighbors-in-depth.md
SOURCES += 5-trees-ensembles-in-depth.md
SOURCES += 5-demo-adaboost.md
SOURCES += 5-knn-tree-bias-variance.md
SOURCES += 5-hw-voter-classification.md
SOURCES += 6-svm-pre-kernel.md
SOURCES += 6-demo-digits-classifiers.md
SOURCES += 7-svm-bias-variance.md
SOURCES += 7-demo-backprop.md
SOURCES += 7-neural-net-demo-draw.md

# Convert the list of source files (Markdown files )
# into a list of output files
NBS := $(patsubst %.md,%.ipynb,$(SOURCES))

# The ones to also execute and generate PDFs
EXECS := 1-python-numpy-tutorial.md
EXECS += 1-exploratory-data-analysis.md
EXECS += 1-data-detective.md
EXECS += 2-compute-by-hand.md
EXECS += 2-regression-r2.md
EXECS += 2-linear-regression-deep-dive.md
EXECS += 2-linear-regression-case-study.md
EXECS += 3-gradient-descent-deep-dive.md
EXECS += 2-advertising-hw.md
EXECS += 3-bias-variance-model-selection-deep-dive.md
EXECS += 3-regularization-deep-dive.md
EXECS += 4-logistic-regression-in-depth.md
EXECS += 4-logistic-regression-digits.md
EXECS += 4-compas-case-study.md
EXECS += 5-k-nearest-neighbors-in-depth.md
EXECS += 5-trees-ensembles-in-depth.md
EXECS += 5-demo-adaboost.md
EXECS += 6-svm-pre-kernel.md
EXECS += 6-demo-digits-classifiers.md
EXECS += 7-svm-bias-variance.md
EXECS += 7-demo-backprop.md

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

