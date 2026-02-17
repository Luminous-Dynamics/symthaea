# CLI Usage Guide

The `mycelix-desci` command-line interface provides powerful tools for interacting with the Mycelix-DeSci network.

## Installation

Build the CLI from source:

```bash
cargo build --release
```

The binary will be at `target/release/mycelix-desci`. Add it to your PATH or run directly.

## Quick Start

### 1. Initialize Configuration

```bash
mycelix-desci init
```

This creates:
- `.mycelix/config.toml` - Configuration file
- `.mycelix/data/` - Data directory
- `.mycelix/claims/` - Local claim storage

### 2. Upload a Dataset

```bash
mycelix-desci upload \
  --file my_dataset.csv \
  --tier E2 \
  --category genomics \
  --description "CRISPR gene editing results" \
  --provenance "Lab Notebook:2024-001" \
  --license "CC-BY-4.0" \
  --keywords "CRISPR,gene-editing,cancer"
```

Output:
```
Calculating dataset hash...
Hash: blake3:abc123def456...

✓ Dataset uploaded successfully!
  Claim ID: 550e8400-e29b-41d4-a716-446655440000
  Tier: E2
  Category: genomics
  Hash: blake3:abc123def456...
  Saved to: .mycelix/claims/550e8400-e29b-41d4-a716-446655440000.json
```

### 3. Query Claims

```bash
# Query by category
mycelix-desci query --category longevity

# Query with tier filter
mycelix-desci query --min-tier E3

# Keyword search
mycelix-desci query --keywords "NAD+" --format table

# JSON output
mycelix-desci query --category genomics --format json
```

### 4. Verify a Claim

```bash
# Verify claim metadata
mycelix-desci verify 550e8400-e29b-41d4-a716-446655440000

# Verify with file hash check
mycelix-desci verify 550e8400-e29b-41d4-a716-446655440000 --file my_dataset.csv
```

## Commands

### `init` - Initialize Configuration

Create default configuration and directory structure.

```bash
mycelix-desci init [--output <DIR>]
```

**Options:**
- `--output, -o <DIR>` - Output directory (default: `.mycelix`)

**Example:**
```bash
mycelix-desci init --output ~/.config/mycelix
```

---

### `hash` - Calculate File Hash

Compute cryptographic hash of a file or directory.

```bash
mycelix-desci hash <PATH> [--algorithm <ALGO>]
```

**Arguments:**
- `<PATH>` - File or directory to hash

**Options:**
- `--algorithm, -a <ALGO>` - Hash algorithm: `blake3` (default) or `sha256`

**Example:**
```bash
mycelix-desci hash dataset.csv --algorithm blake3
```

Output:
```
File: dataset.csv
Algorithm: blake3
Hash: 7a8f9c3e2b1d4f6a8c9e2b5d7f1a3c6e8b4d2f9a5c7e1b3d6f8a2c5e9b7d4f1a

Formatted: blake3:7a8f9c3e2b1d4f6a8c9e2b5d7f1a3c6e8b4d2f9a5c7e1b3d6f8a2c5e9b7d4f1a
```

---

### `upload` - Upload Dataset

Upload a dataset and create an epistemic claim.

```bash
mycelix-desci upload <FILE> \
  --tier <TIER> \
  --category <CATEGORY> \
  --description <DESC> \
  [OPTIONS]
```

**Arguments:**
- `<FILE>` - Dataset file to upload

**Required Options:**
- `--tier, -t <TIER>` - Epistemic tier: `E0`, `E1`, `E2`, `E3`, or `E4`
- `--category, -c <CAT>` - Category (e.g., genomics, longevity, climate)
- `--description, -d <DESC>` - Description of the dataset

**Optional:**
- `--provenance, -p <PROV>` - Provenance information
- `--license, -L <LIC>` - License (e.g., CC-BY-4.0, MIT)
- `--keywords, -k <KW>` - Comma-separated keywords

**Example:**
```bash
mycelix-desci upload mouse_longevity.csv \
  --tier E3 \
  --category longevity \
  --description "NAD+ supplementation effects on mouse lifespan" \
  --provenance "Lab:MIT-Bio-2024" \
  --license "CC-BY-4.0" \
  --keywords "NAD+,aging,mice,lifespan"
```

---

### `verify` - Verify Claim

Verify claim metadata and optionally check file hash.

```bash
mycelix-desci verify <CLAIM_ID> [--file <FILE>]
```

**Arguments:**
- `<CLAIM_ID>` - UUID of the claim to verify

**Options:**
- `--file, -f <FILE>` - Dataset file for hash verification

**Example:**
```bash
# Verify metadata only
mycelix-desci verify 550e8400-e29b-41d4-a716-446655440000

# Verify with file
mycelix-desci verify 550e8400-e29b-41d4-a716-446655440000 --file dataset.csv
```

Output:
```
Claim Information:
  ID: 550e8400-e29b-41d4-a716-446655440000
  Tier: E3
  Category: longevity
  Description: NAD+ supplementation effects...
  Hash: blake3:abc123...
  Verifications: 3

Tier Validation:
  Required verifications: 3
  Current verifications: 3
  Status: ✓ Valid

Hash Verification:
  ✓ Hash matches!
    File: dataset.csv
    Hash: abc123...
```

---

### `query` - Query Claims

Search for claims with filters.

```bash
mycelix-desci query [OPTIONS]
```

**Options:**
- `--category, -c <CAT>` - Filter by category
- `--min-tier, -t <TIER>` - Minimum epistemic tier
- `--keywords, -k <KW>` - Keyword search
- `--format, -f <FMT>` - Output format: `table` (default), `json`, or `text`
- `--limit, -n <N>` - Maximum results (default: 10)

**Examples:**
```bash
# Find longevity research
mycelix-desci query --category longevity

# Find reproducible claims
mycelix-desci query --min-tier E3

# Keyword search
mycelix-desci query --keywords "CRISPR,cancer"

# JSON output for 20 results
mycelix-desci query --category genomics --format json --limit 20
```

Table output:
```
Found 3 claims:

ID                                   Tier   Category        Description
----------------------------------------------------------------------------------------------------
550e8400-e29b-41d4-a716-44665544...  E3     longevity       NAD+ supplementation effects...
660f9511-f3ac-52e5-b827-55776655...  E4     longevity       Resveratrol clinical trial...
770g0622-g4bd-63f6-c938-66887766...  E2     longevity       Caloric restriction in primates
```

---

### `info` - Display Claim Details

Show detailed information about a claim.

```bash
mycelix-desci info <CLAIM_ID> [--format <FMT>]
```

**Arguments:**
- `<CLAIM_ID>` - Claim UUID

**Options:**
- `--format, -f <FMT>` - Output format: `text` (default) or `json`

**Example:**
```bash
mycelix-desci info 550e8400-e29b-41d4-a716-446655440000
```

Output:
```
Claim Information
================================================================================
ID: 550e8400-e29b-41d4-a716-446655440000
Epistemic Tier: E3 (Reproducible with documented methodology)
Category: longevity
Description: NAD+ supplementation effects on mouse lifespan

Dataset:
  Hash: blake3:abc123def456...
  Storage: ipfs://QmX8Y9Z...
  Reproducibility: 92.00%
  License: CC-BY-4.0

Keywords:
  - NAD+
  - aging
  - mice
  - lifespan

Metadata:
  Creator: did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK
  Created: 2025-10-15T14:30:00Z
  Updated: 2025-11-01T10:20:00Z

Verification:
  Verifications: 3 / 3 required
  Status: ✓ Valid

Verifiers:
  1. did:key:z6MkpTHR8VNsBxYAAWHut2Geadd9jSwuBV8xRoAnwWsdvktH (2025-10-20T11:00:00Z)
     Notes: Verified experimental methodology and statistical analysis
  ...

Provenance Chain:
  1. Lab Notebook ID:2024-042 (laboratory_record)
     URL: https://lab.example.edu/notebook/2024-042
     Timestamp: 2025-10-15T14:30:00Z
  ...
```

---

### `config` - Configuration Management

Manage configuration settings.

```bash
mycelix-desci config <ACTION>
```

**Subcommands:**
- `show` - Display current configuration
- `validate` - Validate configuration

**Examples:**
```bash
# Show configuration
mycelix-desci config show

# Validate
mycelix-desci config validate
```

---

## Global Options

These options work with any command:

- `--config, -c <FILE>` - Configuration file path
- `--log-level, -l <LEVEL>` - Log level: `debug`, `info`, `warn`, `error`
- `--help, -h` - Show help
- `--version, -V` - Show version

**Examples:**
```bash
# Use custom config
mycelix-desci --config ~/my-config.toml query --category genomics

# Debug logging
mycelix-desci --log-level debug upload --file data.csv ...

# Show help
mycelix-desci --help
mycelix-desci upload --help
```

---

## Environment Variables

Override configuration with environment variables:

- `MYCELIX_LOG_LEVEL` - Log level
- `MYCELIX_STORAGE_BACKEND` - Storage backend (memory, ipfs, filecoin)
- `MYCELIX_IPFS_API_URL` - IPFS API URL
- `MYCELIX_PORT` - P2P port

**Example:**
```bash
MYCELIX_LOG_LEVEL=debug mycelix-desci query --category longevity
```

---

## Workflows

### Complete Dataset Upload Workflow

```bash
# 1. Initialize
mycelix-desci init

# 2. Calculate hash (optional, for verification)
mycelix-desci hash dataset.csv

# 3. Upload
mycelix-desci upload dataset.csv \
  --tier E2 \
  --category genomics \
  --description "My research data" \
  --license "CC-BY-4.0"

# 4. Note the Claim ID from output

# 5. Verify
mycelix-desci verify <CLAIM_ID> --file dataset.csv

# 6. Query to find it
mycelix-desci query --category genomics
```

### Research Discovery Workflow

```bash
# Find all longevity research
mycelix-desci query --category longevity --min-tier E3

# Get details on interesting claim
mycelix-desci info <CLAIM_ID>

# Export as JSON for analysis
mycelix-desci query --category longevity --format json > longevity_claims.json
```

---

## Tips & Best Practices

1. **Always verify hashes** when downloading datasets:
   ```bash
   mycelix-desci verify <CLAIM_ID> --file downloaded_file.csv
   ```

2. **Use high epistemic tiers** for important research (E3+)

3. **Add detailed provenance** to increase trust:
   ```bash
   --provenance "DOI:10.1234/example,Lab:MIT-2024"
   ```

4. **Tag with keywords** for discoverability:
   ```bash
   --keywords "keyword1,keyword2,keyword3"
   ```

5. **Check tier requirements** before claiming high tiers - E3 needs 3 verifications, E4 needs 5

---

## Troubleshooting

**Problem**: Command not found
**Solution**: Ensure `mycelix-desci` is in your PATH or use full path

**Problem**: Claim not found
**Solution**: Check that `.mycelix/claims/` directory exists and contains the claim file

**Problem**: Hash mismatch
**Solution**: File has been modified; obtain original file from storage ref

**Problem**: Invalid configuration
**Solution**: Run `mycelix-desci config validate` to check errors

---

## Next Steps

- Read [Architecture Guide](architecture.md) for system design
- See [Federated Learning Guide](federated-learning.md) for FL workflows
- Check [Integration Guide](integrations.md) for DeSci platform connections