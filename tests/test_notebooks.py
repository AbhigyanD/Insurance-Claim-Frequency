"""Notebooks must be executed, error-free, and must not misuse the held-out test split."""
import re
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NAMES = ["01_data_exploration", "02_frequency_modeling", "03_severity_modeling", "04_model_evaluation"]
OUTCOMES = ("ClaimNb", "TotalClaimAmount", "ClaimAmount")


def violations(name: str, code: str) -> list[str]:
    """Rules for one code cell. Returns a list of human-readable problems (empty = fine)."""
    problems = []
    code = "\n".join(re.sub(r"#.*", "", line) for line in code.splitlines())     # comments are not code
    touches_test = bool(re.search(r"policies_test|[\"']test[\"']|evaluate_test|test_pred|predictions\.parquet", code))
    if name.startswith("01"):
        # exploration may look at test FEATURES (to compare distributions) but never its outcomes
        if touches_test or re.search(r"policies_\{n\}", code):
            if "columns=" not in code:
                problems.append("reads test data without restricting columns")
            if any(o in code for o in OUTCOMES):
                problems.append("touches test outcomes in exploration")
    elif name.startswith(("02", "03")):
        if touches_test:
            problems.append("modelling notebook references the test split")
    elif name.startswith("04"):
        if "read_parquet" in code or "data/processed" in code or "proc /" in code:
            problems.append("evaluation notebook reads processed data")
        if re.search(r"\.fit\(", code):
            problems.append("evaluation notebook refits a model")
    return problems


def _code_cells(name):
    nb = nbformat.read(NB_DIR / f"{name}.ipynb", as_version=4)
    return nb, [c for c in nb.cells if c.cell_type == "code"]


@pytest.mark.parametrize("name", NAMES)
def test_notebook_is_valid_executed_and_error_free(name):
    nb, cells = _code_cells(name)
    nbformat.validate(nb)
    assert cells and all(c.execution_count is not None for c in cells), "run the notebook so outputs are saved"
    errors = [o for c in cells for o in c.get("outputs", []) if o.output_type == "error"]
    assert not errors, errors[:1]


@pytest.mark.parametrize("name", NAMES)
def test_notebook_respects_holdout_rules(name):
    _, cells = _code_cells(name)
    found = [(i, p) for i, c in enumerate(cells) for p in violations(name, c.source)]
    assert not found, found


def test_rule_checker_catches_violations_and_allows_legitimate_code():
    # the checker itself must flag misuse (otherwise the tests above prove nothing)
    assert violations("02_x", 'test = pd.read_parquet(proc / "policies_test.parquet")')
    assert violations("03_x", "pred.query(\"split == 'test'\")")                  # deliberately conservative: any test reference
    assert violations("03_x", 'df = frame[frame["split"] == "test"]')
    assert violations("03_x", "x = 1  # the test set is not used here") == []      # a comment is not a reference
    assert violations("01_x", 'd = pd.read_parquet(p, columns=["Exposure"])  # no ClaimNb here') == []
    assert violations("01_x", 'd = pd.read_parquet(proc / "policies_test.parquet")')                       # no columns= restriction
    assert violations("01_x", 'd = pd.read_parquet(proc / "policies_test.parquet", columns=["ClaimNb"])')  # outcome column
    assert violations("04_x", "model.fit(X, y)") and violations("04_x", 'pd.read_parquet("data/processed/p.parquet")')
    # legitimate code passes
    assert violations("01_x", 'd = pd.read_parquet(proc / f"policies_{n}.parquet", columns=["Exposure", "DrivAge"])') == []
    assert violations("02_x", 'glm = PoissonGLM(cfg).fit(train)') == []
    assert violations("04_x", 'pd.read_csv(rep / "test_results.csv")') == []
