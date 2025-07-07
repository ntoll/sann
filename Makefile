clean:
	rm -rf build dist *.egg-info
	rm -rf .coverage
	rm -rf .pytest_cache
	rm -rf examples/digit_recognition/nn.json
	rm -rf examples/beep/nn.json
	find . | grep -E "(__pycache__)" | xargs rm -rf

tidy:
	black -l 79 *.py
	black -l 79 examples/digit_recognition/*.py
	black -l 79 examples/beep/*.py

test:
	pytest --cov=sann --cov-report=term-missing

check: clean tidy test

dist: check
	python3 setup.py sdist

publish-test: dist
	twine upload -r test --sign dist/*

publish: dist
	twine upload --sign dist/*