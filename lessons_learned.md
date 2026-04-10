# Lessons Learned

## 2026-04-10: Streamlit Cloud deployment stuck in "preparing"

**Problem:** After rebooting the app on Streamlit Community Cloud, the deployment got stuck in the preparation phase with no error messages.

**Root cause:** The `requirements.txt` had been cleaned up to remove packages not directly imported by `app.py` (`pandas`, `sparklines`, `ipython`). However, these packages are transitive dependencies needed during installation of other packages (e.g., `shap` depends on `pandas`). Without them explicitly listed, Streamlit Cloud's dependency resolver hung silently.

**What we tried (didn't help):**
- Pinning all dependency versions to exact local versions
- Adding a `runtime.txt` to pin Python 3.12
- Rebooting the app multiple times

**What fixed it:** Reverting `requirements.txt` to the original version (commit `1a1c8f9`) that included all packages: `pandas>=2.0.0`, `sparklines>=0.4.2`, `ipython>=8.0.0`.

**Takeaway:** On Streamlit Cloud, always keep all dependencies in `requirements.txt`, even if they seem unused in your code. The platform installs only what is listed, and missing transitive dependencies can cause silent hangs. When debugging deployment issues, compare against the last known working `requirements.txt` first.

---

## 2026-04-10: GitHub HTTPS authentication

**Problem:** `git push` failed with "Invalid username or token. Password authentication is not supported."

**What fixed it:** Using GitHub CLI interactive login:
```bash
gh auth login
# Select: GitHub.com → HTTPS → Login with web browser
# Enter the one-time code at github.com/login/device
```

**Takeaway:** GitHub no longer supports password authentication over HTTPS. Use `gh auth login` for browser-based authentication with a one-time code, or switch to SSH keys.
