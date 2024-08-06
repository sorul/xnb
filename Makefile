flake8:
	@poetry run flake8 --config config/tox.ini

test:
	@poetry run pytest --cov=xnb tests

requirements:
	@poetry export -f requirements.txt --output requirements.txt --without-hashes

dev_requirements:
	@poetry export --with dev -f requirements.txt --output requirements_dev.txt --without-hashes

tag:
	@make flake8
	@make test
	@make requirements
	@make dev_requirements
	@git add .
	@git commit -am "v$$(poetry version -s)"
	@git push
	@git checkout master
	@git merge --no-edit --log developer
	@git tag v$$(poetry version -s)
	@git push
	@git push --tags
	@git checkout developer
	@poetry version