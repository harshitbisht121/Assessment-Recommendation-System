import pandas as pd
from playwright.sync_api import sync_playwright
import time

BASE = "https://www.shl.com/solutions/products/product-catalog/?start="
BASE_DOMAIN = "https://www.shl.com"

def scrape_catalog():
    records = []
    collected_urls = set()  # Track URLs collected from type=1 pages

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        page_size = 12
        
        # Scrape only Individual Test Solutions (type=1)
        product_type = 1
        max_page = 32  # Stop at page 32
        
        print(f"\n{'='*60}")
        print(f"SCRAPING: Individual Test Solutions (Pages 1-{max_page})")
        print(f"{'='*60}")
        
        empty_pages = 0
        page_no = 0
        
        # STEP 1: Collect all product URLs from type=1 catalog pages
        print("\n[STEP 1] Collecting product URLs from Individual Test Solutions pages...")
        
        while page_no < max_page:
            start = page_no * page_size
            url = f"{BASE}{start}&type={product_type}"
            
            print(f"\nScanning catalog page {page_no+1}/{max_page}: {url}")
            
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=60000)
                time.sleep(2)  # Wait for dynamic content
                
                # Get all tables on the page
                tables = page.query_selector_all("table")
                
                if not tables:
                    print("⚠ No tables found")
                    empty_pages += 1
                    if empty_pages >= 3:
                        print("No more pages found. Stopping.")
                        break
                    page_no += 1
                    continue
                
                # Find the table with "Individual Test Solutions" header
                individual_table = None
                for table in tables:
                    try:
                        header = table.query_selector("th.custom__table-heading__title")
                        if header and "Individual Test Solutions" in header.inner_text():
                            individual_table = table
                            break
                    except:
                        continue
                
                if not individual_table:
                    print("⚠ Individual Test Solutions table not found")
                    empty_pages += 1
                    if empty_pages >= 3:
                        print("No more pages found. Stopping.")
                        break
                    page_no += 1
                    continue
                else:
                    empty_pages = 0
                
                # Get all rows from the Individual Test Solutions table only
                table_rows = individual_table.query_selector_all("tbody tr")
                
                page_links = []
                
                # Process each row (skip header row)
                for row in table_rows:
                    try:
                        # Skip header rows
                        if row.query_selector("th"):
                            continue
                        
                        # Find product link in this row
                        link = row.query_selector("a[href*='/products/product-catalog/view/']")
                        
                        if link:
                            href = link.get_attribute("href")
                            if href:
                                if href.startswith("/"):
                                    href = BASE_DOMAIN + href
                                
                                if href not in collected_urls:
                                    collected_urls.add(href)
                                    page_links.append(href)
                    
                    except Exception as e:
                        continue
                
                print(f"✓ Found {len(page_links)} new Individual Test Solutions")
                print(f"  Total collected: {len(collected_urls)}")
                
                page_no += 1
                
            except Exception as e:
                print(f"Error loading page: {e}")
                empty_pages += 1
                if empty_pages >= 3:
                    break
                page_no += 1
        
        # STEP 2: Visit each collected product page
        print(f"\n{'='*60}")
        print(f"[STEP 2] Scraping {len(collected_urls)} Individual Test Solutions...")
        print(f"{'='*60}")
        
        for idx, link in enumerate(sorted(collected_urls), 1):
            try:
                print(f"\n[{idx}/{len(collected_urls)}] Visiting: {link}")
                page.goto(link, wait_until="domcontentloaded", timeout=60000)
                time.sleep(1)
                
                # Extract product name
                try:
                    name = page.locator("h1").first.inner_text().strip()
                except:
                    name = "N/A"
                
                # Extract description from meta tag
                try:
                    desc = page.locator("meta[name='description']").get_attribute("content") or ""
                except:
                    desc = ""
                
                # Get page content to determine test type
                text = page.content().lower()
                
                # Determine test type based on content
                test_types = []
                if "personality" in text or "behavior" in text or "behaviour" in text:
                    test_types.append("P")
                if "cognitive" in text or "ability" in text or "aptitude" in text:
                    test_types.append("A")
                if "knowledge" in text or "skill" in text:
                    test_types.append("K")
                if "simulation" in text:
                    test_types.append("S")
                if "situational judgment" in text or "biodata" in text:
                    test_types.append("B")
                
                # If no type detected, default to K for Individual Tests
                if not test_types:
                    test_types = ["K"]
                
                records.append({
                    "name": name,
                    "url": link,
                    "description": desc,
                    "product_type": "Individual Test Solutions",
                    "duration": 40,
                    "remote_support": "Yes",
                    "adaptive_support": "Yes",
                    "test_type": test_types
                })
                
                print(f"  ✔ {name} - Types: {', '.join(test_types)}")
                
            except Exception as e:
                print(f"  ✘ Failed: {e}")
        
        browser.close()
    
    # Create DataFrame and remove duplicates (just in case)
    df = pd.DataFrame(records).drop_duplicates(subset=["url"])
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Total Individual Test Solutions scraped: {len(df)}")
    
    # Save to CSV
    df.to_csv("data/raw/shl_catalog.csv", index=False)
    print("\n✓ Saved → data/raw/shl_catalog.csv")
    
    return df


if __name__ == "__main__":
    scrape_catalog()