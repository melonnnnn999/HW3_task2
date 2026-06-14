CONFIG ?= configs/task2_act_calvin.yaml
SMOKE_CONFIG ?= configs/task2_act_calvin_smoke.yaml

.PHONY: check-data build-splits train-a train-abc smoke-a smoke-abc eval dry-run dry-run-smoke

check-data:
	python scripts/check_datasets.py --config $(CONFIG)

build-splits:
	python scripts/build_calvin_scene_splits.py

dry-run:
	python scripts/train_act_calvin.py --config $(CONFIG) --run-kind a --dry-run
	python scripts/train_act_calvin.py --config $(CONFIG) --run-kind abc --dry-run

dry-run-smoke:
	python scripts/train_act_calvin.py --config $(SMOKE_CONFIG) --run-kind a --dry-run
	python scripts/train_act_calvin.py --config $(SMOKE_CONFIG) --run-kind abc --dry-run

smoke-a:
	bash scripts/task2_train.sh $(SMOKE_CONFIG) a

smoke-abc:
	bash scripts/task2_train.sh $(SMOKE_CONFIG) abc

train-a:
	bash scripts/task2_train.sh $(CONFIG) a

train-abc:
	bash scripts/task2_train.sh $(CONFIG) abc

eval:
	bash scripts/task2_eval.sh $(CONFIG)
