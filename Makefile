# Makefile — the todo/ gates on the ai-sdlc/tooling branch (item 170.2). The scripts are
# copies of the ai-sdlc deployment repository's todo tooling; FORK.md, section "todo/".
.PHONY: help check-todo todo-index

help: ## List the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/'

check-todo: ## Validate todo/ front matter, ids, depends_on, task ids and TODO.md index drift
	@bash check_todo.sh

todo-index: ## Regenerate the TODO.md index from todo/<id>-<slug>.md front matter
	@bash gen_todo_index.sh
