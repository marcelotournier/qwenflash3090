.PHONY: test start stop status logs qualify bench maya-build maya-download maya-pack

test:
	python3 -m unittest discover -s tests -v
start:
	python3 scripts/server.py start
stop:
	python3 scripts/server.py stop
status:
	python3 scripts/server.py status
logs:
	python3 scripts/server.py logs
qualify:
	python3 scripts/server.py enable-restart
bench:
	python3 scripts/benchmark.py --qualify --concurrency 4
maya-build:
	docker build -f Dockerfile.maya -t qwenflash-maya:local .
maya-download:
	python3 scripts/maya_model.py download
maya-pack:
	python3 scripts/maya_model.py pack
