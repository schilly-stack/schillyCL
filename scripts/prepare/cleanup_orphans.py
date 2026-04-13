from pathlib import Path

clean_dir    = Path("data/crops/clean")
degraded_dir = Path("data/crops/degraded")

degraded_names = {f.name for f in degraded_dir.glob("*.jpg")}

deleted = 0
for clean_file in clean_dir.glob("*.jpg"):
    if clean_file.name not in degraded_names:
        clean_file.unlink()
        print(f"Slettet: {clean_file.name}")
        deleted += 1

print(f"\nFærdig — {deleted} filer slettet")