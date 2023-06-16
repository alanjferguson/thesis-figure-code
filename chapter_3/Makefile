OUT_DIR = ./output/

FIG_PY = $(wildcard fig*.py)

all: pdfs

pdfs: $(FIG_PY)
	$(foreach var,$(FIG_PY), python $(var))

clean:
	rm $(OUT_DIR)/*.pdf
