#!/bin/sh
set -eu

# Runs as root (see Dockerfile). Docker creates named volumes fresh at
# mount time, independent of whatever the image had baked in — observed
# here as both wrong ownership (root, not appuser) *and* a missing write
# bit (mode 555) on the fresh volume, so both have to be fixed at
# container start, not at build time, or the unprivileged app user can't
# write to /app/data at all.
chown -R appuser:appuser /app/data /app/models
chmod -R u+rwX /app/data /app/models

# First boot: real news auto-ingests on demand when configured.
# No synthetic data is seeded.
exec gosu appuser "$@"
