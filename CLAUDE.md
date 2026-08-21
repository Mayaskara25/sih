# sih — project instructions

## Credentials — read this before touching any fetcher

Secrets live at **`~/.config/sih/credentials.env`** (mode 600), deliberately **outside this
repository**. `.env.example` in the repo lists variable *names* only.

**Rules for anyone — human or agent — working in this repo:**

1. **Never `cat`, `head`, `grep`, `echo` or otherwise print `~/.config/sih/credentials.env`,
   and never read it into a message, commit, log or transcript.** An agent that prints it has
   leaked it into a conversation history that outlives the session. There is no need to read it:
   `core.credentials.require()` loads it, and `scripts/check_credentials.py` reports
   configured/not-configured as booleans.
2. **Never hardcode a credential**, not even temporarily while debugging. Use
   `core.credentials.get("CDSE_S3_ACCESS_KEY")` / `require("cdse")`.
3. **Never write credentials into `plan.md`, `docs/`, a notebook, or a Colab/Kaggle cell** (D12
   rule 4 already forbids uploading benchmark data to third-party notebooks; the same applies,
   more strongly, to secrets).
4. If a credential is missing, **stop and tell the user which variable to set.** Do not invent a
   value, do not switch to an unauthenticated endpoint, and do not silently skip the dataset.
5. Exception messages name the **variable**, never the value. Keep it that way.

## Verification standard

Dataset facts in `plan.md` are either **verified against the files** or **documentation-only**,
and `docs/datasets.md` says which. Do not promote a row between tiers without actually opening
the files. HAD100's project page was wrong about its archive in five separate ways (D11); ABU
and HYDICE were wrong in three more (D13). Assume documentation is wrong until checked.

Re-run `scripts/verify_had100.py` and `scripts/verify_benchmarks.py` after any dataset change.
Both exit non-zero on drift.
