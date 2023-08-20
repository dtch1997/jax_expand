import unittest

import jax_expand


class VersionTestCase(unittest.TestCase):
    """ Version tests """

    def test_version(self):
        """ check jax_expand exposes a version attribute """
        self.assertTrue(hasattr(jax_expand, "__version__"))
        self.assertIsInstance(jax_expand.__version__, str)


if __name__ == "__main__":
    unittest.main()
