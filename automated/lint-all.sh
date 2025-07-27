#!/bin/zsh

# Apply a set of lint checks (black, mypy, pylint, ruff) to 
# the calling arguments. Works for Python files and Jupyter 
# notebooks.

# If no arguments provided, find all .py and .ipynb files in current directory
if [ $# -eq 0 ]; then
    set -- $(find . -maxdepth 1 -name "*.py" -o -name "*.ipynb" | sort)
fi

for arg in "$@"
do
    echo "========================================"
    if [[ ! -e "$arg" ]]; then
        echo "File or directory ($arg) not found, skipping ..."
        continue
    fi
    echo "Linting \"$arg\" ..."
    echo "========================================"
    
    # Run ruff on all supported files
    echo -e "\n --- ruff ---"
    ruff format "$arg"
    ruff check --fix "$arg"
    
    # Type checking based on file extension
    if [[ "$arg" == *.ipynb ]]; then
        echo -e "\n$arg is a Jupyter Notebook ..."
        echo -e "\n --- mypy ---"
        nbqa mypy "$arg"
        echo -e "\n --- pyright ---"
        nbqa pyright "$arg"
    elif [[ "$arg" == *.py ]]; then
        echo -e "\n$arg is a Python file ..."
        echo -e "\n --- mypy ---"
        mypy "$arg"
        echo -e "\n --- pyright ---"
        pyright "$arg"
    else
        echo "Argh: $arg file type not supported for type checking, skipping ..."
        continue
    fi
    
    # Check for overrides (works for both .py and .ipynb)
    echo -e "\n\nChecking for type and pylint overrides ..."
    grep "# type: " "$arg"
    grep "# pyright: " "$arg"
    grep "# pylint: " "$arg"
    grep "# noqa: " "$arg"
    grep "# pragma: no cover" "$arg"
    grep --regexp="from typing import .*cast" "$arg"
    grep "cast(" "$arg"
    grep "typing.Any" "$arg"
    grep "@.*overload" "$arg"
    grep "TYPE_CHECKING" "$arg"
done
echo "========================================"
