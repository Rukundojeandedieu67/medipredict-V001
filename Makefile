PYTHON ?= python
ALEMBIC ?= alembic

.PHONY: data reference calibrate acceptance simulate seeds train phase2 test db-up migrate load-reference db-setup

data: reference calibrate

reference:
	$(PYTHON) scripts/prepare_reference.py

simulate: reference
	$(PYTHON) -m medipredict.simulator

calibrate: reference
	$(PYTHON) scripts/calibrate_simulation.py

acceptance: calibrate test
	$(PYTHON) scripts/check_phase1.py

seeds: reference
	$(PYTHON) scripts/generate_seeds.py --count 20

train:
	$(PYTHON) scripts/train_model.py

phase2: seeds
	$(PYTHON) scripts/train_model.py

test:
	$(PYTHON) -m pytest

db-up:
	docker compose up -d db

migrate:
	$(ALEMBIC) upgrade head

load-reference:
	$(PYTHON) -m medipredict.db.load_reference

db-setup: db-up migrate load-reference
