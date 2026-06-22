import os
import json
import subprocess
from pathlib import Path
from datasets import load_dataset
from groq import Groq
from src.config import get_settings

cfg = get_settings()


# ─────────────────────────────────────────────────────────────
# 1.  PUBLIC DATA — SEC 10-K Annual Reports (eloukas/edgar-corpus)
#     HuggingFace dataset with 10-K filings split into sections.
#     We use year_2020 and keep only 30 large-cap companies.
# ─────────────────────────────────────────────────────────────
# Update from a set to a dictionary mapping CIK strings to Ticker symbols
CIK_TO_TICKER = {
    "0000320193": "AAPL",  # Apple
    "0000789019": "MSFT",  # Microsoft
    "0001652044": "GOOGL", # Google
    "0001018724": "AMZN",  # Amazon
    "0001326801": "META",  # Meta
    "0001318605": "TSLA",  # Tesla
    "0001041690": "NVDA",  # Nvidia
    "0000019617": "JPM",   # JPMorgan Chase
    "0000200406": "JNJ",   # Johnson & Johnson
    "0001403161": "V",     # Visa
    "0000080424": "PG",    # Procter & Gamble
    "0000731766": "UNH",   # UnitedHealth
    "0000938443": "HD",    # Home Depot
    "0001141391": "MA",    # Mastercard
    "0001744489": "DIS",   # Disney
    "0000070858": "BAC",   # Bank of America
    "0000796343": "ADBE",  # Adobe
    "0001108524": "CRM",   # Salesforce
    "0001065280": "NFLX",  # Netflix
    "0001633917": "PYPL",  # PayPal
    "0000050863": "INTC",  # Intel
    "0001166691": "CMCSA", # Comcast
    "0000077476": "PEP",   # PepsiCo
    "0000021344": "KO",    # Coca-Cola
    "0000001800": "ABT",   # Abbott Labs
    "0001420800": "TMO",   # Thermo Fisher
    "0000909832": "COST",  # Costco
    "0000320187": "NKE",   # Nike
    "0000310158": "MRK",   # Merck
    "0000104169": "WMT"    # Walmart
}

def download_sec_filings():
    print("⬇ Downloading EDGAR corpus (2020)...")

    dataset = load_dataset(
        "parquet",
        data_files="hf://datasets/eloukas/edgar-corpus@refs/convert/parquet/year_2020/train/*.parquet",
        split="train",
        streaming=True
    )

    output_dir = cfg.data_raw / "public"
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for row in dataset:
        filename_str = row.get("filename", "")
        if not filename_str:
            continue
            
        # Extract the numeric CIK from the filename (e.g., "708821" from "708821_2020.htm")
        raw_cik = filename_str.split("_")[0]
        
        # SEC CIKs are 10 digits long, left-padded with zeros. 
        # Normalize it to match our map (e.g., "708821" -> "0000708821")
        padded_cik = raw_cik.zfill(10)
        
        # Check if this CIK matches one of our targeted companies
        if padded_cik not in CIK_TO_TICKER:
            continue
            
        ticker = CIK_TO_TICKER[padded_cik]

        # Extract textual report sections
        sections = {
            k: v for k, v in row.items()
            if k.startswith("section_") and v and len(str(v)) > 200
        }

        doc = {
            "ticker":   ticker,
            "filename": filename_str,
            "year":     2020,
            "sections": sections,
            "metadata": {
                "tier":   "public",
                "source": "sec_10k",
                "ticker": ticker,
                "year":   2020,
            }
        }

        out_path = output_dir / f"{ticker}_10k_2020.json"
        out_path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        saved += 1
        print(f"  ✓ Saved {ticker} ({saved}/{len(CIK_TO_TICKER)})")

        if saved >= len(CIK_TO_TICKER):
            break

    print(f"\n✅ SEC filings saved: {saved} companies → {output_dir}")


# ─────────────────────────────────────────────────────────────
# 2.  CONFIDENTIAL DATA — Open-source HR handbooks (simulated)
#     Real company handbooks published under open licenses.
#     We label these as "hr" tier to simulate confidential docs.
# ─────────────────────────────────────────────────────────────
HR_REPOS = [
    {"name": "basecamp_handbook", "url": "https://github.com/basecamp/handbook"},
    {"name": "opengovfoundation_hr", "url": "https://github.com/opengovfoundation/hr-manual"},
    {"name": "clef_handbook",      "url": "https://github.com/clef/handbook"},
]

def download_hr_handbooks():
    print("\n⬇ Cloning HR handbooks...")
    hr_dir = cfg.data_raw / "hr"
    hr_dir.mkdir(parents=True, exist_ok=True)

    for repo in HR_REPOS:
        dest = hr_dir / repo["name"]
        if dest.exists():
            print(f"  ⏭ Already exists: {repo['name']}")
            continue
        subprocess.run(["git", "clone", "--depth=1", repo["url"], str(dest)], check=True)
        print(f"  ✓ Cloned: {repo['name']}")

        # Write a metadata sidecar alongside each handbook
        # so the ingestion pipeline knows the access tier
        meta = {
            "source": repo["name"],
            "metadata": {"tier": "hr", "dept": "all", "source": "hr_handbook"}
        }
        (dest / "_metadata.json").write_text(json.dumps(meta))

    print(f"✅ HR handbooks saved → {hr_dir}")


# ─────────────────────────────────────────────────────────────
# 3.  CONFIDENTIAL DATA — Synthetic performance reviews via Groq
#     We generate PII-rich documents to test Presidio detection.
#     These are purely synthetic — no real people involved.
# ─────────────────────────────────────────────────────────────
REVIEW_PROMPT = """Generate a realistic employee performance review.
Include: full name, employee ID (EMP-XXXXX format), email address,
department, manager name, salary band (e.g. L4-$95,000), review period,
ratings (1-5) for 4 competencies, and 2 paragraphs of written feedback.
Return plain text only, no markdown."""

EMPLOYEES = [
    ("Engineering", "Alice Chen"), ("Engineering", "Bob Ramirez"),
    ("Finance", "Carol Smith"),   ("Finance", "David Park"),
    ("HR", "Emma Johnson"),        ("Sales", "Frank Mueller"),
    ("Legal", "Grace Li"),         ("Marketing", "Henry Brown"),
]

def generate_synthetic_reviews():
    print("\n🤖 Generating synthetic performance reviews via Groq...")
    client = Groq(api_key=cfg.groq_api_key)
    conf_dir = cfg.data_raw / "confidential"
    conf_dir.mkdir(parents=True, exist_ok=True)

    for dept, name in EMPLOYEES:
        prompt = f"Department: {dept}. Employee: {name}.\n\n{REVIEW_PROMPT}"

        response = client.chat.completions.create(
            model=cfg.groq_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
        )
        review_text = response.choices[0].message.content

        doc = {
            "employee_name": name,
            "department":    dept,
            "content":       review_text,
            "metadata": {
                "tier":   "confidential",
                "dept":   dept.lower(),
                "source": "performance_review",
                # confidential docs are scoped to department managers
            }
        }
        fname = f"review_{name.replace(' ', '_').lower()}.json"
        (conf_dir / fname).write_text(json.dumps(doc, ensure_ascii=False))
        print(f"  ✓ Generated review for {name} ({dept})")

    print(f"✅ Synthetic reviews saved → {conf_dir}")


if __name__ == "__main__":
    download_sec_filings()
    # download_hr_handbooks()
    # generate_synthetic_reviews()
    print("\n🎉 All datasets downloaded successfully!")