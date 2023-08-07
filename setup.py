
"""setups."""
import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.0.1"

REPO_NAME = "MLOps-Project"
AUTHOR_USERNAME = "Stanlito-AI"
SRC_REPO = "mlops-project"
AUTHOR_EMAIL = "mosesstanley99@gmail.com"

setuptools.setup(
    name = SRC_REPO,
    version = __version__,
    author = AUTHOR_USERNAME,
    author_email = AUTHOR_EMAIL,
    description = "A python package for MLOPs Project",
    long_description = long_description,
    long_description_content = "text/markdown",
    url = f"https://github.com/{AUTHOR_USERNAME}/{REPO_NAME}",
    project_urls = {
        "Bug Tracker": f"https://github.com/{AUTHOR_USERNAME}/{REPO_NAME}/issues",
    },
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where="src")
)