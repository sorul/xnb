flake8:
	@poetry run flake8 --config config/tox.ini

test:
	@poetry run pytest --cov=xnb tests

requirements:
	@poetry lock
	@poetry export -f requirements.txt --output requirements.txt --without-hashes

dev_requirements:
	@poetry lock
	@poetry export --with dev -f requirements.txt --output requirements_dev.txt --without-hashes

push_develop:
	@make requirements
	@make dev_requirements
	@git commit --allow-empty -am "updating requirements and pushing to git"
	@git push

tag:
	@make flake8
	@make test
	@make requirements
	@make dev_requirements
	@git add .
	@git commit -am "v$$(poetry version -s)"
	@git push
	@git merge --no-edit --log developer
	@git tag v$$(poetry version -s)
	@poetry version
	@echo "Tagging complete. Make a pull request to merge developer into master -> https://github.com/sorul/xnb/compare/developer?expand=1"