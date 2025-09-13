def pytest_addoption(parser):
    g = parser.getgroup("pytest")
    g.addoption("--device", choices=["cpu", "cuda"], default="cpu")
    g.addoption("--remote", action="append", default=[])
