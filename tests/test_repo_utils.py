import unittest

from src.repo_utils import parse_repo_url


class RepoUrlParsingTests(unittest.TestCase):
    def test_https_url(self):
        self.assertEqual(parse_repo_url("https://github.com/microsoft/vscode"), ("microsoft", "vscode"))

    def test_without_scheme(self):
        self.assertEqual(parse_repo_url("github.com/microsoft/vscode"), ("microsoft", "vscode"))

    def test_git_ssh_url(self):
        self.assertEqual(parse_repo_url("git@github.com:microsoft/vscode.git"), ("microsoft", "vscode"))

    def test_invalid_url_raises(self):
        with self.assertRaises(ValueError):
            parse_repo_url("https://github.com/microsoft")


if __name__ == "__main__":
    unittest.main()
