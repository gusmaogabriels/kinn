# Releasing the CLI

The distribution installs the `kinn` Python package and `kinn` command. Release
builds use `kinn.__version__`; a release does not require a version change when
that version has never been published for this project.

Merge the reviewed changes into `main` after CI passes. Create the corresponding
`v<version>` tag from that commit, then run the **Publish package** workflow on
that tag. Choose `testpypi` or `pypi` explicitly; ordinary pushes do not publish.
The workflow rejects a tag that does not match the package version, runs the
cross-platform tests and installed-wheel checks, and publishes that verified
wheel and source distribution.

Publishing uses PyPI Trusted Publishing, configured separately for each registry:

| Setting | Value |
| --- | --- |
| Repository owner | `gusmaogabriels` |
| Repository | `kinn` |
| Workflow filename | `publish.yml` |
| GitHub environment | `pypi` (or `testpypi`) |

The PyPI project name must match the distribution name in `pyproject.toml`, and
the publisher must be registered on a project owned by its maintainer. Until
that is configured and an upload succeeds, use the repository or a verified
GitHub release artifact for installation. See the
[PyPI setup instructions](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).

For a GitHub release, attach the wheel, source distribution and a `SHA256SUMS`
file. Give discovery services the immutable commit, artifact URL and checksum,
supported commands and installed-wheel verification results. Advertise PyPI
installation only after its files are available under the correct project.
