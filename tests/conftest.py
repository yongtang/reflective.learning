def pytest_addoption(parser):
    parser.addoption(
        "--cuda",
        action="store_true",
        default=False,
        help="Include CUDA device params in tests (adds device='cuda').",
    )


def pytest_generate_tests(metafunc):
    if "device" not in metafunc.fixturenames:
        return

    devices = ["cpu"]

    if metafunc.config.getoption("--cuda"):
        devices.append("cuda")

    metafunc.parametrize("device", devices)
