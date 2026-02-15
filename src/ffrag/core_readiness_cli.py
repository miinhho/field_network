from __future__ import annotations

import argparse
import io
import unittest


DEFAULT_TEST_MODULES = (
    "tests.test_core_readiness",
    "tests.test_integration_end_to_end",
    "tests.test_integration_extremes",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run core readiness/integration tests and print summary")
    parser.add_argument(
        "--modules",
        type=str,
        default=",".join(DEFAULT_TEST_MODULES),
        help="Comma-separated unittest module paths",
    )
    args = parser.parse_args()

    modules = tuple(m.strip() for m in args.modules.split(",") if m.strip())
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for mod in modules:
        suite.addTests(loader.loadTestsFromName(mod))

    stream = io.StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=1)
    result = runner.run(suite)

    print("module_count,tests_run,failures,errors,successful")
    print(f"{len(modules)},{result.testsRun},{len(result.failures)},{len(result.errors)},{int(result.wasSuccessful())}")
    if not result.wasSuccessful():
        print("---- details ----")
        print(stream.getvalue())


if __name__ == "__main__":
    main()
