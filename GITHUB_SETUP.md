# GitHub Repository Setup for Sparse

## Required GitHub Secrets & Variables

### For HuggingFace Space Deployment

You need to configure these in your GitHub repository settings:

**Settings → Secrets and variables → Actions**

---

## 🔐 Secrets (encrypted)

### 1. `HF_TOKEN`
- **Type:** Repository secret
- **What it is:** HuggingFace access token for deploying to Spaces
- **How to get it:**
  1. Go to https://huggingface.co/settings/tokens
  2. Create a new token with **write** permissions
  3. Copy the token
  4. In GitHub: Settings → Secrets and variables → Actions → New repository secret
  5. Name: `HF_TOKEN`
  6. Value: Paste your HF token

**Why needed:** Allows GitHub Actions to push demo to your HuggingFace Space

---

## 📝 Variables (plain text)

### 1. `HF_SPACE_ID`
- **Type:** Repository variable
- **What it is:** Your HuggingFace Space identifier
- **Format:** `username/space-name`
- **Example:** `gagansuie/sparse-demo` or `sparselabs/sparse`
- **How to set:**
  1. In GitHub: Settings → Secrets and variables → Actions → Variables tab
  2. Click "New repository variable"
  3. Name: `HF_SPACE_ID`
  4. Value: `your-username/your-space-name`

**Why needed:** Tells GitHub Actions where to deploy the demo

---

## 🎯 Step-by-Step Setup

### Step 1: Create HuggingFace Space

1. Go to https://huggingface.co/new-space
2. **Space name:** `sparse-demo` (or your choice)
3. **License:** Choose appropriate license
4. **Space SDK:** Gradio
5. **Space hardware:** CPU Basic (free)
6. Click "Create Space"

Your Space URL will be: `https://huggingface.co/spaces/[your-username]/sparse-demo`

### Step 2: Get HuggingFace Token

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. **Name:** `github-actions-sparse`
4. **Type:** Write
5. Click "Generate"
6. **Copy the token immediately** (you won't see it again)

### Step 3: Add Secrets to GitHub

1. Go to your GitHub repo: https://github.com/gagansuie/sparse
2. Click **Settings** (top menu)
3. In left sidebar: **Secrets and variables** → **Actions**
4. Click **"New repository secret"**
5. Name: `HF_TOKEN`
6. Value: Paste your HuggingFace token
7. Click **"Add secret"**

### Step 4: Add Variables to GitHub

1. Still in Settings → Secrets and variables → Actions
2. Click the **"Variables"** tab
3. Click **"New repository variable"**
4. Name: `HF_SPACE_ID`
5. Value: `gagansuie/sparse-demo` (use your actual username/space-name)
6. Click **"Add variable"**

---

## ✅ Verify Setup

After adding secrets and variables, you can test the deployment:

### Option 1: Manual Trigger
1. Go to **Actions** tab in GitHub
2. Click **"Deploy Public Demo to HF Space"** workflow
3. Click **"Run workflow"** dropdown
4. Click **"Run workflow"** button

### Option 2: Automatic Trigger
Push any change to `hf_space/` directory:
```bash
cd hf_space/
touch .trigger  # Create dummy file
git add .
git commit -m "Test HF Space deployment"
git push
```

Check the **Actions** tab to see the deployment progress.

---

## 🚀 Release Process

When you tag a release, the build workflow creates deliverables:

```bash
# Create and push tag
git tag v0.2.0
git push origin v0.2.0
```

This will:
1. ✅ Run all tests
2. ✅ Build Python wheels
3. ✅ Create deliverables tarball
4. ✅ Upload as GitHub draft release
5. ⏸️ Wait for your manual approval to publish

To publish the release:
1. Go to **Releases** tab
2. Find the draft release
3. Click **"Edit"**
4. Review the deliverables
5. Click **"Publish release"**

---

## 📦 Deliverables Package Contents

When you tag `v0.2.0`, the CI creates:

```
sparse-v0.2.0-deliverables.tar.gz
├── Source code (core/, optimizer/, cli/)
├── Pre-built wheels (dist/)
├── Documentation (docs/)
├── Examples (examples/)
├── Tests (tests/)
└── DELIVERABLES.md (manifest)
```

With SHA256 checksum: `sparse-v0.2.0-deliverables.tar.gz.sha256`

---

## 🔒 Security Best Practices

1. ✅ **Repository set to PRIVATE** (prevents IP leaks)
2. ✅ **HF Space demo uses mock data only** (no proprietary code)
3. ✅ **Releases are draft by default** (manual approval required)
4. ✅ **Proprietary LICENSE** (prevents open-sourcing)
5. ✅ **90-day artifact retention** (limited client access window)

---

## 🛠️ Troubleshooting

### Error: "HF_TOKEN not found"
- Check that you added it as a **secret** (not variable)
- Make sure the secret name is exactly `HF_TOKEN` (case-sensitive)

### Error: "HF_SPACE_ID not set"
- Check that you added it as a **variable** (not secret)
- Make sure the variable name is exactly `HF_SPACE_ID`
- Format must be `username/space-name`

### Error: "Permission denied" when pushing to HF Space
- Your HF token needs **write** permissions
- Regenerate token at https://huggingface.co/settings/tokens
- Update the `HF_TOKEN` secret in GitHub

### Workflow doesn't trigger
- Check that you're pushing to `main` branch
- For HF Space: changes must be in `hf_space/` directory
- Check Actions tab for any disabled workflows

---

## 📧 Questions?

Contact: gagan.suie@sparselabs.ai

---

**Setup complete! Your CI/CD is ready for proprietary product delivery.** 🎉
