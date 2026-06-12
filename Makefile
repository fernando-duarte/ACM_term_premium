PYTHON ?= python3
OFFICIAL ?= https://www.newyorkfed.org/medialibrary/media/research/data_indicators/ACMTermPremium.xls
REPRO_OUTPUT ?= outputs/ACMTermPremium_reproduced.xlsx
UPDATE_OUTPUT ?= outputs/ACMTermPremium_updated.xlsx
MAX_ABS_DIFF_BP ?= 0.01

.PHONY: reproduce update verify lint format clean distclean

reproduce:
	$(PYTHON) reproduce_acm.py --official "$(OFFICIAL)" --output "$(REPRO_OUTPUT)"

update:
	$(PYTHON) reproduce_acm.py --output "$(UPDATE_OUTPUT)"

verify:
	$(PYTHON) reproduce_acm.py --official "$(OFFICIAL)" --output "$(REPRO_OUTPUT)" --refresh --assert-official-reproduced --max-abs-diff-bp $(MAX_ABS_DIFF_BP)

lint:
	$(PYTHON) -m pre_commit run --all-files

format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

lint:
	$(PYTHON) -m pre_commit run --all-files

format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

clean:
	rm -rf outputs data_cache __pycache__

distclean: clean
	@if [ -d .venv ]; then \
		chmod -R u+w .venv 2>/dev/null || true; \
		find .venv -name .DS_Store -delete 2>/dev/null || true; \
		rm -rf .venv; \
		if [ -d .venv ]; then \
			sleep 1; \
			find .venv -name .DS_Store -delete 2>/dev/null || true; \
			rm -rf .venv; \
		fi; \
		if [ -d .venv ]; then \
			echo "Could not remove .venv. Run 'deactivate' and then retry make distclean."; \
			exit 1; \
		fi; \
	fi
