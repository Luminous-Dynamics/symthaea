# Runnable reference

This folder contains the pure-Python research reference for MCR² v17.

Run from this directory:
~~~bash
python3 run_benchmark.py
python3 -m unittest discover -v
~~~

The generator and regression suite use only the Python standard library. To generate a bounded smoke corpus and validate its two separate schemas:
~~~bash
python3 run_benchmark.py --blocks 64 --write-data --output report-smoke.json
python3 validate_schema_samples.py
~~~

The schema validator requires python-jsonschema. The repository workflow runs these commands on pull requests. This is not a production controller, survival estimator, causal estimator, simulator or hardware adapter.