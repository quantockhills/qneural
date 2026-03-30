# CI/CD Setup for qneural

**Date**: March 30, 2026
**Status**: ✅ Complete - Basic CI configured

---

## What Was Set Up

### 1. GitHub Actions CI Workflow ✅

**File**: `.github/workflows/tests.yml`

**What it does**:
- Runs automatically on every push to `main` or `develop` branches
- Runs on every pull request
- Tests on Python 3.9, 3.10, 3.11, 3.12
- Runs only **fast tests** (skips 26 slow integration tests)

**Command used**: `pytest -v -m "not slow" --tb=short`

**Runtime**: ~2-3 minutes (vs 30-120 minutes for full suite)

---

### 2. Pytest Configuration ✅

**File**: `pytest.ini`

**Custom markers defined**:
- `@pytest.mark.slow` - Marks slow tests (deselect with `-m "not slow"`)
- `@pytest.mark.integration` - Marks integration tests
- `@pytest.mark.unit` - Marks unit tests

**Tests marked as slow** (26 tests across 9 files):
1. `tests/integration/test_autodiff_through_ode.py` (3 tests)
2. `tests/integration/test_batched_time_optimal.py` (2 tests)
3. `tests/integration/test_cz_convergence.py` (4 tests)
4. `tests/integration/test_cz_gate_optimization.py` (3 tests)
5. `tests/integration/test_high_fidelity_training.py` (2 tests)
6. `tests/integration/test_minimal_training.py` (1 test)
7. `tests/integration/test_nn_to_ode_connection.py` (5 tests)
8. `tests/integration/test_ode_method_speed.py` (1 test)
9. `tests/integration/test_training_pipeline.py` (5 tests)

**Physics validation tests NOT marked as slow**:
- `tests/integration/test_physics_validation.py` (19 tests) - These run fast (~7 sec) ✅

---

### 3. README Updates ✅

**Badges added**:
- ![Tests](https://github.com/quantockhills/qneural/workflows/Tests/badge.svg) - Shows CI status
- ![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)
- ![License](https://img.shields.io/badge/license-MIT-green)
- ![Status](https://img.shields.io/badge/status-beta-yellow)

**Testing section added**:
- Clear statement about beta status
- Test pass rates
- Recommendations for production vs early adopters

---

## How to Use Locally

### Run only fast tests (what CI runs):
```bash
pytest -m "not slow"
```

### Run only slow tests:
```bash
pytest -m "slow"
```

### Run all tests (including slow):
```bash
pytest
```

### Run specific test categories:
```bash
pytest -m "integration"  # All integration tests
pytest -m "unit"         # All unit tests
```

### Run tests on specific Python version:
```bash
# Using tox (if configured)
tox -e py39
tox -e py310
```

---

## What Happens on GitHub

### When you push code:

1. **GitHub detects the push**
2. **Triggers the workflow** (`.github/workflows/tests.yml`)
3. **Spins up 4 virtual machines** (one for each Python version)
4. **Each machine**:
   - Checks out your code
   - Installs Python (3.9, 3.10, 3.11, or 3.12)
   - Installs dependencies (`pip install -e ".[dev]"`)
   - Runs fast tests (`pytest -m "not slow"`)
5. **Reports results**:
   - ✅ Green checkmark if all pass
   - ❌ Red X if any fail
   - Badge on README updates automatically

### Timeline:
- Setup: ~30 seconds per Python version
- Test run: ~2-3 minutes per Python version
- Total: ~3-4 minutes for all 4 Python versions (run in parallel)

---

## What's NOT Set Up Yet (For v1.0)

### CD (Continuous Deployment):
- [ ] Automatic PyPI publishing on release tags
- [ ] Automatic documentation building
- [ ] Docker image creation

### Extended CI:
- [ ] Code coverage reporting (pytest-cov)
- [ ] Linting (ruff, black, mypy)
- [ ] Security scanning (bandit, safety)
- [ ] Performance benchmarks

### Workflow file for CD (future):
`.github/workflows/publish.yml` - Will auto-publish to PyPI when you create a release

---

## Next Steps

### Before First Push to GitHub:

1. **Verify GitHub repo URL** in README badges:
   - Current: `quantockhills/qneural`
   - Make sure this matches your actual GitHub username/repo

2. **Test locally first**:
   ```bash
   pytest -m "not slow" -v
   ```

3. **Commit and push**:
   ```bash
   git add .github/workflows/tests.yml pytest.ini README.md
   git add tests/integration/*.py  # Updated with markers
   git commit -m "feat: Add CI/CD with GitHub Actions

   - Add pytest.ini with slow/integration markers
   - Mark 26 slow integration tests to skip in CI
   - Add GitHub Actions workflow for fast tests
   - Update README with badges and test status

   CI runs fast tests only (~3 min) on Python 3.9-3.12"
   git push
   ```

4. **Check GitHub Actions tab**:
   - Go to https://github.com/quantockhills/qneural/actions
   - Watch the workflow run
   - See if tests pass ✅

### If Tests Fail in CI:

**Common issues**:
1. **Missing dependencies** - Check `setup.py` has all requirements
2. **Path issues** - CI runs from repo root
3. **Environment differences** - CI uses fresh Ubuntu, not your conda env

**Fix**:
- Check the GitHub Actions logs
- Reproduce locally: `pytest -m "not slow"`
- Fix and push again

---

## Statistics

### Tests in CI (Fast):
- **~160 tests** run in CI
- **Runtime**: 2-3 minutes
- **Pass rate**: ~90% (166/185 from last local run)

### Tests NOT in CI (Slow):
- **26 tests** skipped in CI
- **Runtime**: 30-120 minutes
- **Reason**: Training, optimization, benchmarks
- **Status**: Run locally before major releases

---

## Benefits

### ✅ What you get:
1. **Automatic testing** on every push
2. **Multi-Python support** (3.9-3.12) tested automatically
3. **Professional appearance** (badges, CI status)
4. **Catch bugs early** before merging PRs
5. **Free for public repos** (GitHub Actions included)

### ⚠️ Limitations:
1. Slow tests not run (by design)
2. Only runs on Ubuntu (can add macOS/Windows later)
3. 2000 minutes/month free tier (plenty for this project)

---

## Resources

### GitHub Actions:
- [Documentation](https://docs.github.com/en/actions)
- [Workflow syntax](https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions)
- [Python with GitHub Actions](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python)

### Pytest Markers:
- [Marking test functions](https://docs.pytest.org/en/stable/example/markers.html)
- [Skip and xfail](https://docs.pytest.org/en/stable/how-to/skipping.html)

---

**Setup completed**: March 30, 2026
**Next review**: Before v0.6 release (add CD for PyPI)

