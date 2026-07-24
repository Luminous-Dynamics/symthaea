# Patch 0018: feat retirement decommission mutation tools and endpoints

**Series:** 25

## Objective

Make operational tooling honor archive-only mode and remove accidental mutation surfaces.

## Intended changes

- Add terminal checks to CLIs, service routes, background jobs, and administrative workflows.
- Disable or remove signing, publishing, recovery, reopening, allowance, and policy-rotation endpoints for the retired identity.
- Keep verification and export commands available.

## Required tests

- Endpoint inventory confirms every mutation route is blocked.
- Cached sessions and queued jobs cannot commit after retirement.
- Read-only commands remain functional.

## Non-claims

- Does not remotely erase third-party software.
- Does not prove external operators stopped running old binaries.
